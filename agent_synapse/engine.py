"""
Unified Agent-Synapse Runtime Engine.

Coordinates:
- Stage 1: Early-Layer Attention Entropy & Logit-Lens Tool Predictor
- Stage 2: Speculative FastMCP Wire Pipeline (with Idempotency Safety Gate)
- Stage 3: Zero-Penalty SRAM KV-Cache Micro-Checkpointing & AST Equivalence Splicer
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from agent_synapse.kv_checkpoint import SRAMKVCacheCheckpointer
from agent_synapse.models import (
    AcceleratorProfile,
    AttentionHeadTelemetry,
    IdempotencyClass,
    LogitLensProjection,
    SpeculativeExecutionMetrics,
    ToolSchemaSpec,
)
from agent_synapse.predictor import EarlyLayerLogitLensPredictor
from agent_synapse.speculative_wire import SpeculativeFastMCPWirePipeline


class AgentSynapseEngine:
    """
    End-to-end speculative tool prefetching and SRAM KV-cache checkpointing engine.
    Implements Theorem 3's closed-form compute-I/O overlap latency reduction:
      T_synapse = T_decode + max(0, T_io - T_lead)           (on speculative AST hit)
      T_synapse = T_decode + T_rollback + T_io               (on speculative mismatch)
    """

    def __init__(
        self,
        tool_registry: Dict[str, ToolSchemaSpec],
        tau_conf: float = 0.82,
        accelerator_profile: Optional[AcceleratorProfile] = None,
    ):
        self.accelerator = accelerator_profile or AcceleratorProfile.tpu_v6e()
        self.tool_registry = dict(tool_registry)
        self.predictor = EarlyLayerLogitLensPredictor(
            tau_conf=tau_conf,
            accelerator_profile=self.accelerator,
        )
        self.wire_pipeline = SpeculativeFastMCPWirePipeline(tool_registry=self.tool_registry)
        self.checkpointer = SRAMKVCacheCheckpointer(accelerator_profile=self.accelerator)
        self.metrics_history: List[SpeculativeExecutionMetrics] = []

    def execute_turn(
        self,
        trace_id: str,
        attention_telemetry: Sequence[AttentionHeadTelemetry],
        logit_projections: Sequence[LogitLensProjection],
        actual_emitted_tool_id: str,
        actual_emitted_args: Dict[str, Any],
        branch_token_index: int = 128,
        total_decode_ms: float = 310.0,
        override_io_latency_ms: Optional[float] = None,
    ) -> SpeculativeExecutionMetrics:
        """
        Executes a complete autoregressive tool-calling turn with Agent-Synapse:
        1. Runs Stage 1 early-layer prediction at decode step t_branch.
        2. Pins an on-chip SRAM CoW KV-cache micro-checkpoint.
        3. Dispatches Stage 2 speculative FastMCP prefetch if conformal & idempotent gates pass.
        4. At <tool_call> EOS, runs Stage 3 canonical AST equivalence check:
           splices prefetched payload on hit, or rolls back SRAM CoW pointer in < 12 us on miss.
        """
        prediction = self.predictor.predict_from_early_layers(
            attention_telemetry=attention_telemetry,
            logit_projections=logit_projections,
            registered_tools=self.tool_registry,
        )

        checkpoint = self.checkpointer.create_micro_checkpoint(
            branch_token_index=branch_token_index,
            speculative_span_tokens=28,
        )

        actual_spec = self.tool_registry.get(actual_emitted_tool_id)
        io_latency_ms = (
            override_io_latency_ms
            if override_io_latency_ms is not None
            else (actual_spec.mean_io_latency_ms if actual_spec else 240.0)
        )
        idemp_class = (
            actual_spec.idempotency_class
            if actual_spec is not None
            else IdempotencyClass.MUTATING_BLOCKED
        )

        envelope = self.wire_pipeline.dispatch_speculative_prefetch(
            prediction=prediction,
            checkpoint_hash=checkpoint.checkpoint_hash,
            simulated_io_latency_ms=io_latency_ms,
        )

        spec_dispatched = envelope.status in ("PREFETCHED_OK", "DRY_RUN_SANDBOXED")
        baseline_serial_ms = round(total_decode_ms + io_latency_ms, 3)

        if spec_dispatched:
            ast_res = self.checkpointer.splice_or_rollback_at_eos(
                checkpoint_id=checkpoint.checkpoint_id,
                predicted_tool_id=prediction.predicted_tool_id,
                predicted_args=prediction.predicted_args,
                actual_tool_id=actual_emitted_tool_id,
                actual_args=actual_emitted_args,
                prefetched_payload=envelope.response_payload,
            )
            if ast_res.is_equivalent:
                # Theorem 3: Compute-I/O overlap hides min(T_io, T_lead)
                residual_io_ms = max(0.0, io_latency_ms - prediction.estimated_lead_time_ms)
                synapse_ms = round(
                    total_decode_ms + residual_io_ms + (ast_res.sram_operation_latency_us / 1000.0),
                    3,
                )
                rollback_triggered = False
            else:
                # Mismatch: < 12 us CoW rollback + normal post-EOS tool execution
                synapse_ms = round(
                    total_decode_ms + (ast_res.sram_operation_latency_us / 1000.0) + io_latency_ms,
                    3,
                )
                rollback_triggered = True
            sram_us = ast_res.sram_operation_latency_us
            ast_hit = ast_res.is_equivalent
        else:
            # Not speculatively prefetched (either below tau_conf or blocked non-idempotent)
            synapse_ms = baseline_serial_ms
            sram_us = 0.0
            ast_hit = False
            rollback_triggered = False

        metrics = SpeculativeExecutionMetrics(
            trace_id=trace_id,
            tool_id=actual_emitted_tool_id,
            idempotency_class=idemp_class,
            conformal_confidence=prediction.conformal_confidence,
            speculative_dispatched=spec_dispatched,
            ast_equivalent_hit=ast_hit,
            rollback_triggered=rollback_triggered,
            decode_lead_time_ms=prediction.estimated_lead_time_ms if spec_dispatched else 0.0,
            backend_io_latency_ms=io_latency_ms,
            baseline_serial_latency_ms=baseline_serial_ms,
            synapse_effective_latency_ms=synapse_ms,
            latency_saved_ms=round(baseline_serial_ms - synapse_ms, 3),
            sram_checkpoint_rollback_us=sram_us,
            wire_headers_emitted=dict(envelope.headers),
        )
        self.metrics_history.append(metrics)
        return metrics
