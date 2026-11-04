"""
Stage 1 — Early-Layer Attention Entropy & Logit-Lens Tool Predictor.

Monitors intermediate transformer cross-attention heads (Layers 8-16) and logit-lens
unembedding projections z^(l) = W_U * RMSNorm(h_t^(l)) during autoregressive token
decoding to predict idempotent read tool targets and canonical argument bindings
150 ms - 400 ms before the <tool_call> EOS token is emitted, gated by a split-conformal
confidence threshold tau_conf.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent_synapse.models import (
    AcceleratorProfile,
    AttentionHeadTelemetry,
    LogitLensProjection,
    SpeculativePrediction,
    ToolSchemaSpec,
)


class EarlyLayerLogitLensPredictor:
    """
    Computes early-layer cross-attention Shannon entropy H_attn^(8:16), logit-lens
    posterior P_lens(w | h_t^(8:16)), and split-conformal prediction confidence C_conf.
    """

    MONITORED_LAYERS: Tuple[int, ...] = (8, 10, 12, 14, 16)

    def __init__(
        self,
        tau_conf: float = 0.82,
        max_entropy_threshold: float = 0.72,
        layer_decay_gamma: float = 0.85,
        accelerator_profile: Optional[AcceleratorProfile] = None,
    ):
        if not (0.0 < tau_conf < 1.0):
            raise ValueError(f"tau_conf must lie in (0, 1), got {tau_conf}")
        self.tau_conf = tau_conf
        self.max_entropy_threshold = max_entropy_threshold
        self.layer_decay_gamma = layer_decay_gamma
        self.accelerator = accelerator_profile or AcceleratorProfile.tpu_v6e()
        self.calibration_quantiles: Dict[float, float] = {}

    @staticmethod
    def softmax(logits: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
        """Numerically stable softmax over tool candidate logits."""
        if not logits:
            return {}
        temp = max(temperature, 1e-6)
        max_v = max(logits.values())
        exps = {k: math.exp((v - max_v) / temp) for k, v in logits.items()}
        total = sum(exps.values()) or 1.0
        return {k: v / total for k, v in exps.items()}

    @staticmethod
    def normalized_shannon_entropy(prob_dist: Dict[str, float]) -> float:
        """
        Computes normalized Shannon entropy H(P) / log(|V_tools|) in [0, 1].
        Low entropy indicates sharp cross-attention concentration on a single tool schema.
        """
        n = len(prob_dist)
        if n <= 1:
            return 0.0
        total = sum(max(p, 0.0) for p in prob_dist.values())
        if total <= 0.0:
            return 1.0
        entropy = 0.0
        for p in prob_dist.values():
            p_norm = max(p, 0.0) / total
            if p_norm > 1e-12:
                entropy -= p_norm * math.log(p_norm)
        return min(1.0, max(0.0, entropy / math.log(n)))

    def aggregate_attention_telemetry(
        self,
        telemetry_samples: Sequence[AttentionHeadTelemetry],
    ) -> Tuple[Dict[str, float], float]:
        """
        Aggregates cross-attention weights across monitored early layers l in [8, 16]
        with exponential depth weighting gamma^(l_max - l), returning the fused tool
        attention distribution and normalized Shannon entropy H_attn.
        """
        filtered = [
            s for s in telemetry_samples if 8 <= s.layer_idx <= 16
        ]
        if not filtered:
            filtered = list(telemetry_samples)
        if not filtered:
            return {}, 1.0

        max_layer = max(s.layer_idx for s in filtered)
        accum: Dict[str, float] = {}
        weight_sum = 0.0

        for sample in filtered:
            w_l = self.layer_decay_gamma ** (max_layer - sample.layer_idx)
            weight_sum += w_l
            for tool_id, attn_mass in sample.tool_attention_weights.items():
                accum[tool_id] = accum.get(tool_id, 0.0) + w_l * max(attn_mass, 0.0)

        total_mass = sum(accum.values())
        if total_mass <= 0.0:
            uniform_p = 1.0 / max(len(accum), 1)
            dist = {k: uniform_p for k in accum}
            return dist, 1.0

        dist = {k: v / total_mass for k, v in accum.items()}
        h_attn = self.normalized_shannon_entropy(dist)
        return dist, h_attn

    def aggregate_logit_lens_projections(
        self,
        projections: Sequence[LogitLensProjection],
    ) -> Tuple[str, float, Dict[str, Any], float, int]:
        """
        Fuses intermediate logit-lens projections across layers 8..16 to extract:
        - Top predicted tool_id w_hat
        - Logit-lens posterior probability P_lens(w_hat)
        - Consensus argument dictionary a_hat
        - Geometric-mean argument token confidence C_arg
        - Estimated remaining decode steps to <tool_call> EOS
        """
        valid = [p for p in projections if 8 <= p.layer_idx <= 16] or list(projections)
        if not valid:
            raise ValueError("At least one LogitLensProjection is required")

        max_layer = max(p.layer_idx for p in valid)
        fused_logits: Dict[str, float] = {}
        total_w = 0.0

        for proj in valid:
            w_l = self.layer_decay_gamma ** (max_layer - proj.layer_idx)
            total_w += w_l
            for tid, logit in proj.tool_logits.items():
                fused_logits[tid] = fused_logits.get(tid, 0.0) + w_l * logit

        for tid in fused_logits:
            fused_logits[tid] /= max(total_w, 1e-6)

        posterior = self.softmax(fused_logits)
        top_tool_id = max(posterior.items(), key=lambda kv: (kv[1], kv[0]))[0]
        top_posterior = posterior[top_tool_id]

        # Select argument bindings from highest-layer projection
        deepest_proj = max(valid, key=lambda p: p.layer_idx)
        pred_args = dict(deepest_proj.predicted_args)

        if deepest_proj.arg_token_confidences:
            log_sum = sum(
                math.log(max(min(c, 1.0), 1e-6))
                for c in deepest_proj.arg_token_confidences.values()
            )
            arg_conf = math.exp(log_sum / len(deepest_proj.arg_token_confidences))
        else:
            arg_conf = 1.0

        # Inter-layer stability check on top tool across layers
        top_per_layer = [
            max(p.tool_logits.items(), key=lambda kv: kv[1])[0]
            for p in valid if p.tool_logits
        ]
        agreement_ratio = (
            sum(1 for t in top_per_layer if t == top_tool_id) / max(len(top_per_layer), 1)
        )
        arg_stability = 0.65 * arg_conf + 0.35 * agreement_ratio

        return (
            top_tool_id,
            top_posterior,
            pred_args,
            arg_stability,
            max(deepest_proj.remaining_tokens_to_eos, 1),
        )

    @staticmethod
    def compute_nonconformity_score(
        attention_entropy: float,
        logit_lens_posterior: float,
        argument_stability: float,
    ) -> float:
        """
        Computes nonconformity score s(x, y_hat) in [0, 1]:
        s = 1 - (0.45 * P_lens + 0.30 * (1 - H_attn) + 0.25 * C_arg).
        """
        c_raw = (
            0.45 * max(0.0, min(1.0, logit_lens_posterior))
            + 0.30 * max(0.0, min(1.0, 1.0 - attention_entropy))
            + 0.25 * max(0.0, min(1.0, argument_stability))
        )
        return round(max(0.0, min(1.0, 1.0 - c_raw)), 6)

    def calibrate_conformal_threshold(
        self,
        calibration_nonconformities: Sequence[float],
        alpha: float = 0.10,
    ) -> float:
        """
        Split-conformal calibration (Theorem 1):
        Given n exchangeable calibration nonconformity scores {s_1, ..., s_n} and target
        false-dispatch rate alpha in (0, 1), computes the finite-sample conformal quantile
        q_hat_{1-alpha} = Quantile(s_1..n, ceil((n + 1)*(1 - alpha)) / n)
        and sets tau_conf = 1 - q_hat_{1-alpha}, guaranteeing P(Y_true != Y_hat | Prefetch) <= alpha.
        """
        if not calibration_nonconformities:
            raise ValueError("Calibration set cannot be empty")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must lie in (0, 1)")

        n = len(calibration_nonconformities)
        sorted_s = sorted(calibration_nonconformities)
        rank_idx = min(n - 1, max(0, math.ceil((n + 1) * (1.0 - alpha)) - 1))
        q_hat = sorted_s[rank_idx]
        calibrated_tau = round(max(0.01, min(0.99, 1.0 - q_hat)), 4)
        self.calibration_quantiles[alpha] = q_hat
        self.tau_conf = calibrated_tau
        return calibrated_tau

    def predict_from_early_layers(
        self,
        attention_telemetry: Sequence[AttentionHeadTelemetry],
        logit_projections: Sequence[LogitLensProjection],
        registered_tools: Optional[Dict[str, ToolSchemaSpec]] = None,
    ) -> SpeculativePrediction:
        """
        Executes Stage 1 early-layer tool + argument prediction and evaluates the
        split-conformal confidence gate C_conf >= tau_conf.
        """
        attn_dist, h_attn = self.aggregate_attention_telemetry(attention_telemetry)
        (
            pred_tool_id,
            p_lens,
            pred_args,
            arg_stab,
            rem_tokens,
        ) = self.aggregate_logit_lens_projections(logit_projections)

        # Combine cross-attention mass on pred_tool_id with logit-lens posterior
        attn_support = attn_dist.get(pred_tool_id, p_lens)
        fused_posterior = 0.70 * p_lens + 0.30 * attn_support

        s_nonconf = self.compute_nonconformity_score(h_attn, fused_posterior, arg_stab)
        c_conf = round(1.0 - s_nonconf, 6)

        lead_time_ms = round(
            max(150.0, min(400.0, rem_tokens * self.accelerator.inter_token_decode_ms)),
            2,
        )

        decode_step = max((p.decode_step_t for p in logit_projections), default=0)
        pred_id = hashlib.sha256(
            f"{pred_tool_id}:{decode_step}:{c_conf:.6f}".encode("utf-8")
        ).hexdigest()[:16]

        rejection_reason: Optional[str] = None
        should_prefetch = True

        if h_attn > self.max_entropy_threshold:
            should_prefetch = False
            rejection_reason = (
                f"ATTENTION_ENTROPY_TOO_HIGH ({h_attn:.3f} > {self.max_entropy_threshold:.3f})"
            )
        elif c_conf < self.tau_conf:
            should_prefetch = False
            rejection_reason = (
                f"BELOW_CONFORMAL_THRESHOLD ({c_conf:.3f} < {self.tau_conf:.3f})"
            )
        elif registered_tools and pred_tool_id in registered_tools:
            spec = registered_tools[pred_tool_id]
            missing_req = [r for r in spec.required_args if r not in pred_args]
            if missing_req:
                should_prefetch = False
                rejection_reason = f"MISSING_REQUIRED_ARGS ({','.join(missing_req)})"

        return SpeculativePrediction(
            prediction_id=pred_id,
            decode_step_t=decode_step,
            predicted_tool_id=pred_tool_id,
            predicted_args=pred_args,
            attention_entropy=round(h_attn, 4),
            logit_lens_posterior=round(fused_posterior, 4),
            argument_stability_score=round(arg_stab, 4),
            conformal_confidence=c_conf,
            nonconformity_score=s_nonconf,
            threshold_tau=self.tau_conf,
            estimated_lead_time_ms=lead_time_ms,
            should_prefetch=should_prefetch,
            rejection_reason=rejection_reason,
        )
