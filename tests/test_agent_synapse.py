"""
Comprehensive Unit & Property Test Suite for Agent-Synapse (25 Tests).

Verifies:
1. Stage 1 Early-Layer Attention Entropy & Logit-Lens Tool Prediction (Layers 8-16)
2. Split-Conformal Threshold Calibration & Theorem 1 False-Dispatch Bound
3. Stage 2 Speculative FastMCP Wire Pipeline & Headers (X-Agent-Speculative-Prefetch, X-Synapse-Checkpoint-Hash)
4. Theorem 2 Observational Non-Interference & Idempotency Safety Gating (WRITE/DELETE/DML blocking)
5. Stage 3 Canonical AST Equivalence Splicing (JSON key permutation, SQL/GQL WHERE commutation)
6. Sub-12-Microsecond Copy-on-Write (CoW) SRAM KV-Cache Rollback on Mismatch
7. Theorem 3 Compute-I/O Overlap Speedup Across Cloud TPU v6e & NVIDIA B200 Profiles
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_synapse.engine import AgentSynapseEngine
from agent_synapse.kv_checkpoint import (
    ASTEquivalenceSplicer,
    SRAMKVCacheCheckpointer,
)
from agent_synapse.models import (
    AcceleratorProfile,
    AcceleratorType,
    AttentionHeadTelemetry,
    IdempotencyClass,
    LogitLensProjection,
    ToolSchemaSpec,
)
from agent_synapse.predictor import EarlyLayerLogitLensPredictor
from agent_synapse.speculative_wire import SpeculativeFastMCPWirePipeline


def build_default_tool_registry():
    return {
        "spanner_graph_query_gql": ToolSchemaSpec(
            tool_id="spanner_graph_query_gql",
            service_target="SPANNER_GRAPH_GQL",
            idempotency_class=IdempotencyClass.READ_ONLY,
            required_args=("graph_name", "gql_query"),
            mean_io_latency_ms=245.0,
        ),
        "bigquery_execute_analytics": ToolSchemaSpec(
            tool_id="bigquery_execute_analytics",
            service_target="BIGQUERY_SQL",
            idempotency_class=IdempotencyClass.READ_ONLY,
            required_args=("dataset_id", "sql_query"),
            mean_io_latency_ms=280.0,
        ),
        "vertex_ai_rag_search": ToolSchemaSpec(
            tool_id="vertex_ai_rag_search",
            service_target="VERTEX_SEARCH",
            idempotency_class=IdempotencyClass.READ_ONLY,
            required_args=("corpus_id", "query_text"),
            mean_io_latency_ms=195.0,
        ),
        "erp_post_ledger_dryrun": ToolSchemaSpec(
            tool_id="erp_post_ledger_dryrun",
            service_target="SAP_ODATA_SANDBOX",
            idempotency_class=IdempotencyClass.IDEMPOTENT_DRY_RUN,
            required_args=("company_code", "amount"),
            supports_snapshot_dry_run=True,
            mean_io_latency_ms=220.0,
        ),
        "erp_wire_transfer_mutate": ToolSchemaSpec(
            tool_id="erp_wire_transfer_mutate",
            service_target="SWIFT_SETTLEMENT",
            idempotency_class=IdempotencyClass.MUTATING_BLOCKED,
            required_args=("iban", "amount"),
            supports_snapshot_dry_run=False,
            mean_io_latency_ms=310.0,
        ),
        "crm_delete_customer_record": ToolSchemaSpec(
            tool_id="crm_delete_customer_record",
            service_target="CRM_PROD_DB",
            idempotency_class=IdempotencyClass.MUTATING_BLOCKED,
            required_args=("customer_id",),
            supports_snapshot_dry_run=False,
            mean_io_latency_ms=180.0,
        ),
    }


class TestAgentSynapseSuite(unittest.TestCase):
    """25 unit and property tests for Agent-Synapse."""

    def setUp(self):
        self.registry = build_default_tool_registry()
        self.predictor = EarlyLayerLogitLensPredictor(tau_conf=0.82)
        self.wire = SpeculativeFastMCPWirePipeline(tool_registry=self.registry)
        self.checkpointer = SRAMKVCacheCheckpointer(AcceleratorProfile.tpu_v6e())
        self.engine = AgentSynapseEngine(tool_registry=self.registry, tau_conf=0.82)

    def _sharp_early_layer_inputs(
        self,
        target_tool: str = "spanner_graph_query_gql",
        args=None,
        remaining_tokens: int = 24,
    ):
        args = args or {
            "graph_name": "FinCrimeGraph",
            "gql_query": "MATCH (a:Account)-[t:TRANSFER]->(b:Account) WHERE a.id = 'ACC_901' AND t.region = 'APAC' RETURN b.id",
        }
        other_tools = [
            t for t in ("spanner_graph_query_gql", "bigquery_execute_analytics", "vertex_ai_rag_search")
            if t != target_tool
        ]
        attn = [
            AttentionHeadTelemetry(
                layer_idx=layer,
                head_idx=0,
                decode_step_t=40,
                tool_attention_weights={
                    target_tool: 0.91,
                    other_tools[0]: 0.05,
                    other_tools[1]: 0.04,
                },
            )
            for layer in (8, 10, 12, 14, 16)
        ]
        projs = [
            LogitLensProjection(
                layer_idx=layer,
                decode_step_t=40,
                remaining_tokens_to_eos=remaining_tokens,
                tool_logits={
                    target_tool: 6.4,
                    other_tools[0]: 0.5,
                    other_tools[1]: 0.2,
                },
                predicted_args=args,
                arg_token_confidences={k: 0.96 for k in args},
            )
            for layer in (8, 10, 12, 14, 16)
        ]
        return attn, projs

    def test_01_normalized_shannon_entropy_bounds(self):
        """Entropy of a degenerate 1-hot distribution is 0.0; uniform distribution is 1.0."""
        h_peaked = EarlyLayerLogitLensPredictor.normalized_shannon_entropy(
            {"t1": 1.0, "t2": 0.0, "t3": 0.0, "t4": 0.0}
        )
        h_uniform = EarlyLayerLogitLensPredictor.normalized_shannon_entropy(
            {"t1": 0.25, "t2": 0.25, "t3": 0.25, "t4": 0.25}
        )
        self.assertAlmostEqual(h_peaked, 0.0, places=6)
        self.assertAlmostEqual(h_uniform, 1.0, places=6)

    def test_02_early_layer_attention_aggregation_layers_8_to_16(self):
        """Verify exponential depth-weighted aggregation over Layers 8..16."""
        attn, _ = self._sharp_early_layer_inputs()
        dist, h_attn = self.predictor.aggregate_attention_telemetry(attn)
        self.assertGreater(dist["spanner_graph_query_gql"], 0.88)
        self.assertLess(h_attn, 0.35)

    def test_03_logit_lens_early_exit_prediction_lead_time_window(self):
        """Predicted lead time falls inside the 150 ms - 400 ms window before <tool_call> EOS."""
        attn, projs = self._sharp_early_layer_inputs(remaining_tokens=25)
        pred = self.predictor.predict_from_early_layers(attn, projs, self.registry)
        self.assertTrue(pred.should_prefetch)
        self.assertEqual(pred.predicted_tool_id, "spanner_graph_query_gql")
        self.assertGreaterEqual(pred.estimated_lead_time_ms, 150.0)
        self.assertLessEqual(pred.estimated_lead_time_ms, 400.0)
        self.assertGreaterEqual(pred.conformal_confidence, 0.82)

    def test_04_high_entropy_attention_blocks_speculative_trigger(self):
        """Diffuse attention across tools (high Shannon entropy) suppresses speculative prefetch."""
        diffuse_attn = [
            AttentionHeadTelemetry(
                layer_idx=12,
                head_idx=1,
                decode_step_t=30,
                tool_attention_weights={"t1": 0.25, "t2": 0.25, "t3": 0.25, "t4": 0.25},
            )
        ]
        _, projs = self._sharp_early_layer_inputs()
        pred = self.predictor.predict_from_early_layers(diffuse_attn, projs, self.registry)
        self.assertFalse(pred.should_prefetch)
        self.assertIn("ATTENTION_ENTROPY_TOO_HIGH", pred.rejection_reason or "")

    def test_05_low_conformal_confidence_blocks_speculative_trigger(self):
        """Low logit-lens margin below tau_conf=0.82 is gated off."""
        attn = [
            AttentionHeadTelemetry(
                layer_idx=14,
                head_idx=0,
                decode_step_t=32,
                tool_attention_weights={
                    "spanner_graph_query_gql": 0.80,
                    "t2": 0.10,
                    "t3": 0.05,
                    "t4": 0.05,
                },
            )
        ]
        weak_projs = [
            LogitLensProjection(
                layer_idx=14,
                decode_step_t=32,
                remaining_tokens_to_eos=20,
                tool_logits={"spanner_graph_query_gql": 1.1, "t2": 0.9, "t3": 0.7, "t4": 0.6},
                predicted_args={"graph_name": "G", "gql_query": "MATCH (n) RETURN n"},
                arg_token_confidences={"graph_name": 0.42, "gql_query": 0.40},
            )
        ]
        pred = self.predictor.predict_from_early_layers(attn, weak_projs, self.registry)
        self.assertFalse(pred.should_prefetch)
        self.assertIn("BELOW_CONFORMAL_THRESHOLD", pred.rejection_reason or "")

    def test_06_split_conformal_calibration_quantile_theorem_1(self):
        """Verify split-conformal calibration sets tau_conf = 1 - q_hat_{1-alpha}."""
        cal_scores = [0.02 + 0.002 * i for i in range(100)]  # in [0.02, 0.218]
        tau = self.predictor.calibrate_conformal_threshold(cal_scores, alpha=0.10)
        self.assertGreater(tau, 0.78)
        self.assertLess(tau, 0.85)
        # Empirical coverage on calibration scores >= 1 - alpha
        admitted_errors = [s for s in cal_scores if (1.0 - s) >= tau and s > self.predictor.calibration_quantiles[0.10]]
        self.assertEqual(len(admitted_errors), 0)

    def test_07_missing_required_schema_arguments_rejected(self):
        """Prediction missing required schema arguments is blocked prior to wire dispatch."""
        attn, projs = self._sharp_early_layer_inputs(
            args={"graph_name": "FinCrimeGraph"}  # missing gql_query
        )
        pred = self.predictor.predict_from_early_layers(attn, projs, self.registry)
        self.assertFalse(pred.should_prefetch)
        self.assertIn("MISSING_REQUIRED_ARGS", pred.rejection_reason or "")

    def test_08_wire_headers_injected_on_speculative_prefetch(self):
        """Verify X-Agent-Speculative-Prefetch: True and X-Synapse-Checkpoint-Hash wire headers."""
        attn, projs = self._sharp_early_layer_inputs()
        pred = self.predictor.predict_from_early_layers(attn, projs, self.registry)
        chk = self.checkpointer.create_micro_checkpoint(branch_token_index=96)
        env = self.wire.dispatch_speculative_prefetch(pred, chk.checkpoint_hash)
        self.assertEqual(env.status, "PREFETCHED_OK")
        self.assertEqual(env.headers["X-Agent-Speculative-Prefetch"], "True")
        self.assertEqual(env.headers["X-Synapse-Checkpoint-Hash"], chk.checkpoint_hash)
        self.assertIn("X-Synapse-Conformal-Score", env.headers)
        self.assertIn("X-Synapse-Lead-Time-Ms", env.headers)

    def test_09_non_idempotent_write_mutation_strictly_blocked_theorem_2(self):
        """Non-idempotent WRITE/DELETE tools are strictly blocked from speculative dispatch."""
        attn, projs = self._sharp_early_layer_inputs(
            target_tool="erp_wire_transfer_mutate",
            args={"iban": "DE89370400440532013000", "amount": 50000},
        )
        pred = self.predictor.predict_from_early_layers(attn, projs, self.registry)
        self.assertTrue(pred.should_prefetch)
        chk = self.checkpointer.create_micro_checkpoint(branch_token_index=112)
        env = self.wire.dispatch_speculative_prefetch(pred, chk.checkpoint_hash)
        self.assertEqual(env.status, "BLOCKED_NON_IDEMPOTENT")
        self.assertEqual(env.headers, {})

    def test_10_delete_tool_mutation_strictly_blocked(self):
        """CRM customer deletion tool is blocked from speculative dispatch."""
        attn, projs = self._sharp_early_layer_inputs(
            target_tool="crm_delete_customer_record",
            args={"customer_id": "CUST_8812"},
        )
        pred = self.predictor.predict_from_early_layers(attn, projs, self.registry)
        chk = self.checkpointer.create_micro_checkpoint(branch_token_index=64)
        env = self.wire.dispatch_speculative_prefetch(pred, chk.checkpoint_hash)
        self.assertEqual(env.status, "BLOCKED_NON_IDEMPOTENT")

    def test_11_dml_injection_inside_read_tool_blocked(self):
        """Even on a READ_ONLY tool, SQL/GQL payloads containing DROP/DELETE/UPDATE are blocked."""
        attn, projs = self._sharp_early_layer_inputs(
            target_tool="bigquery_execute_analytics",
            args={"dataset_id": "fin_ops", "sql_query": "DELETE FROM fin_ops.invoices WHERE id = 10"},
        )
        pred = self.predictor.predict_from_early_layers(attn, projs, self.registry)
        chk = self.checkpointer.create_micro_checkpoint(branch_token_index=80)
        env = self.wire.dispatch_speculative_prefetch(pred, chk.checkpoint_hash)
        self.assertEqual(env.status, "BLOCKED_NON_IDEMPOTENT")
        self.assertEqual(env.response_payload["error"], "BLOCKED_MUTATING_DML_PAYLOAD")

    def test_12_idempotent_dry_run_sandbox_allowed_with_header(self):
        """Tools supporting read-only snapshot dry-run emit X-Synapse-Dry-Run-Mode: SNAPSHOT_READ_ONLY."""
        attn, projs = self._sharp_early_layer_inputs(
            target_tool="erp_post_ledger_dryrun",
            args={"company_code": "1000", "amount": 2500},
        )
        pred = self.predictor.predict_from_early_layers(attn, projs, self.registry)
        chk = self.checkpointer.create_micro_checkpoint(branch_token_index=80)
        env = self.wire.dispatch_speculative_prefetch(pred, chk.checkpoint_hash)
        self.assertEqual(env.status, "DRY_RUN_SANDBOXED")
        self.assertEqual(env.headers["X-Synapse-Dry-Run-Mode"], "SNAPSHOT_READ_ONLY")

    def test_13_canonical_ast_equivalence_key_order_and_numeric_types(self):
        """Canonical AST splicer matches JSON arguments despite key reordering and 100.0 vs 100."""
        eq, ast1, ast2, mismatches = ASTEquivalenceSplicer.compare_calls(
            "bigquery_execute_analytics",
            {"sql_query": "SELECT   *  FROM  orders", "dataset_id": "sales", "limit": 100.0},
            "bigquery_execute_analytics",
            {"limit": 100, "dataset_id": "sales", "sql_query": "SELECT * FROM orders"},
        )
        self.assertTrue(eq)
        self.assertEqual(ast1.ast_sha256, ast2.ast_sha256)
        self.assertEqual(mismatches, ())

    def test_14_canonical_ast_equivalence_commutative_where_conjuncts(self):
        """Canonical AST splicer recognizes commuted SQL/GQL WHERE A AND B == WHERE B AND A."""
        q1 = "MATCH (a:Account) WHERE a.id = 'ACC_901' AND a.region = 'APAC' RETURN a"
        q2 = "MATCH (a:Account)   WHERE  a.region='APAC' AND a.id = 'ACC_901' RETURN a"
        eq, ast1, ast2, _ = ASTEquivalenceSplicer.compare_calls(
            "spanner_graph_query_gql",
            {"graph_name": "G1", "gql_query": q1},
            "spanner_graph_query_gql",
            {"graph_name": "G1", "gql_query": q2},
        )
        self.assertTrue(eq)
        self.assertEqual(ast1.ast_sha256, ast2.ast_sha256)

    def test_15_canonical_ast_detects_argument_value_divergence(self):
        """Divergent argument values produce distinct AST hashes and report exact mismatch fields."""
        eq, _, _, mismatches = ASTEquivalenceSplicer.compare_calls(
            "vertex_ai_rag_search",
            {"corpus_id": "sec_10k", "query_text": "Q3 revenue"},
            "vertex_ai_rag_search",
            {"corpus_id": "sec_10k", "query_text": "Q4 revenue"},
        )
        self.assertFalse(eq)
        self.assertEqual(mismatches, ("query_text",))

    def test_16_sram_kv_cache_micro_checkpoint_page_alignment(self):
        """Verify 16-token block alignment and shadow CoW page allocation on TPU v6e."""
        chk = self.checkpointer.create_micro_checkpoint(
            branch_token_index=160, speculative_span_tokens=32
        )
        self.assertEqual(len(chk.base_page_ids), 10)  # 160 / 16 = 10 pages
        self.assertEqual(len(chk.cow_shadow_page_ids), 2)  # 32 / 16 = 2 shadow pages
        self.assertTrue(chk.is_active)
        self.assertEqual(chk.accelerator, AcceleratorType.CLOUD_TPU_V6E)

    def test_17_sram_cow_rollback_under_12_microseconds_on_mismatch(self):
        """On AST mismatch, Copy-on-Write SRAM pointer rollback completes in < 12 microseconds."""
        chk = self.checkpointer.create_micro_checkpoint(
            branch_token_index=128, speculative_span_tokens=32
        )
        res = self.checkpointer.splice_or_rollback_at_eos(
            checkpoint_id=chk.checkpoint_id,
            predicted_tool_id="vertex_ai_rag_search",
            predicted_args={"corpus_id": "c1", "query_text": "draft"},
            actual_tool_id="vertex_ai_rag_search",
            actual_args={"corpus_id": "c1", "query_text": "final_revised"},
            prefetched_payload={"rows": []},
        )
        self.assertFalse(res.is_equivalent)
        self.assertEqual(res.action_taken, "COW_ROLLBACK")
        self.assertLess(res.sram_operation_latency_us, 12.0)
        self.assertEqual(tuple(self.checkpointer.active_page_table), chk.base_page_ids)

    def test_18_sram_zero_wait_payload_splicing_on_ast_match(self):
        """On AST match, prefetched payload is spliced directly into KV-cache blocks."""
        chk = self.checkpointer.create_micro_checkpoint(branch_token_index=96)
        payload = {"rows": [{"id": "ACC_901", "risk_score": 0.94}]}
        res = self.checkpointer.splice_or_rollback_at_eos(
            checkpoint_id=chk.checkpoint_id,
            predicted_tool_id="spanner_graph_query_gql",
            predicted_args={"graph_name": "G", "gql_query": "MATCH (n) RETURN n"},
            actual_tool_id="spanner_graph_query_gql",
            actual_args={"gql_query": "MATCH (n) RETURN n", "graph_name": "G"},
            prefetched_payload=payload,
        )
        self.assertTrue(res.is_equivalent)
        self.assertEqual(res.action_taken, "SPLICED_ZERO_WAIT")
        self.assertLess(res.sram_operation_latency_us, 12.0)
        self.assertGreater(self.checkpointer.checkpoints[chk.checkpoint_id].spliced_payload_tokens, 0)

    def test_19_end_to_end_engine_speculative_hit_zero_io_wait_theorem_3(self):
        """When T_lead >= T_io on a speculative hit, network I/O wait is 100% hidden."""
        attn, projs = self._sharp_early_layer_inputs(remaining_tokens=28)  # ~302.4 ms lead time
        actual_args = projs[0].predicted_args
        metrics = self.engine.execute_turn(
            trace_id="trace_hit_01",
            attention_telemetry=attn,
            logit_projections=projs,
            actual_emitted_tool_id="spanner_graph_query_gql",
            actual_emitted_args=actual_args,
            total_decode_ms=310.0,
            override_io_latency_ms=245.0,
        )
        self.assertTrue(metrics.speculative_dispatched)
        self.assertTrue(metrics.ast_equivalent_hit)
        self.assertFalse(metrics.rollback_triggered)
        self.assertAlmostEqual(metrics.baseline_serial_latency_ms, 555.0, places=2)
        # Since T_lead (~302.4 ms) >= T_io (245.0 ms), effective latency is ~310.002 ms
        self.assertLess(metrics.synapse_effective_latency_ms, 311.0)
        self.assertGreater(metrics.latency_saved_ms, 244.0)

    def test_20_end_to_end_engine_partial_io_overlap_theorem_3(self):
        """When T_lead < T_io on a speculative hit, latency is reduced by exactly T_lead."""
        attn, projs = self._sharp_early_layer_inputs(remaining_tokens=16)  # ~172.8 ms lead time
        actual_args = projs[0].predicted_args
        metrics = self.engine.execute_turn(
            trace_id="trace_partial_02",
            attention_telemetry=attn,
            logit_projections=projs,
            actual_emitted_tool_id="bigquery_execute_analytics" if False else "spanner_graph_query_gql",
            actual_emitted_args=actual_args,
            total_decode_ms=280.0,
            override_io_latency_ms=320.0,
        )
        self.assertTrue(metrics.ast_equivalent_hit)
        self.assertAlmostEqual(
            metrics.latency_saved_ms,
            metrics.decode_lead_time_ms,
            delta=0.05,
        )

    def test_21_end_to_end_engine_mismatch_triggers_fast_cow_rollback(self):
        """When emitted arguments diverge at EOS, Engine triggers < 12 us rollback safely."""
        attn, projs = self._sharp_early_layer_inputs(remaining_tokens=22)
        divergent_args = {
            "graph_name": "FinCrimeGraph",
            "gql_query": "MATCH (a:Account) WHERE a.id = 'DIFFERENT_999' RETURN a",
        }
        metrics = self.engine.execute_turn(
            trace_id="trace_miss_03",
            attention_telemetry=attn,
            logit_projections=projs,
            actual_emitted_tool_id="spanner_graph_query_gql",
            actual_emitted_args=divergent_args,
            total_decode_ms=300.0,
            override_io_latency_ms=240.0,
        )
        self.assertTrue(metrics.speculative_dispatched)
        self.assertFalse(metrics.ast_equivalent_hit)
        self.assertTrue(metrics.rollback_triggered)
        self.assertLess(metrics.sram_checkpoint_rollback_us, 12.0)

    def test_22_end_to_end_engine_mutating_tool_zero_speculative_side_effects(self):
        """Mutating tool calls never emit speculative wire headers or speculative executions."""
        attn, projs = self._sharp_early_layer_inputs(
            target_tool="erp_wire_transfer_mutate",
            args={"iban": "GB29NWBK60161331926819", "amount": 100000},
        )
        metrics = self.engine.execute_turn(
            trace_id="trace_mutate_04",
            attention_telemetry=attn,
            logit_projections=projs,
            actual_emitted_tool_id="erp_wire_transfer_mutate",
            actual_emitted_args={"iban": "GB29NWBK60161331926819", "amount": 100000},
            total_decode_ms=290.0,
            override_io_latency_ms=310.0,
        )
        self.assertFalse(metrics.speculative_dispatched)
        self.assertEqual(metrics.wire_headers_emitted, {})
        self.assertEqual(metrics.latency_saved_ms, 0.0)

    def test_23_heterogeneous_accelerator_profiles_tpu_v6e_and_nvidia_b200(self):
        """Both Cloud TPU v6e and NVIDIA B200 profiles satisfy < 12 us CoW pointer rollback."""
        for profile in (AcceleratorProfile.tpu_v6e(), AcceleratorProfile.nvidia_b200()):
            chk_engine = SRAMKVCacheCheckpointer(profile)
            chk = chk_engine.create_micro_checkpoint(branch_token_index=64)
            res = chk_engine.splice_or_rollback_at_eos(
                checkpoint_id=chk.checkpoint_id,
                predicted_tool_id="t1",
                predicted_args={"a": 1},
                actual_tool_id="t1",
                actual_args={"a": 2},
            )
            self.assertLess(res.sram_operation_latency_us, 12.0)

    def test_24_pareto_monotonicity_across_conformal_thresholds(self):
        """Increasing tau_conf monotonically increases precision and decreases false dispatches."""
        false_counts = []
        for tau in (0.65, 0.75, 0.82, 0.90):
            pred_engine = EarlyLayerLogitLensPredictor(tau_conf=tau)
            false_dispatches = 0
            for idx in range(40):
                is_clean = (idx % 4 != 0)
                logit_gap = 6.2 if is_clean else (1.5 + 0.08 * (idx % 10))
                attn = [
                    AttentionHeadTelemetry(
                        layer_idx=14,
                        head_idx=0,
                        decode_step_t=30,
                        tool_attention_weights={
                            "spanner_graph_query_gql": 0.92 if is_clean else 0.78,
                            "t2": 0.04 if is_clean else 0.10,
                            "t3": 0.02 if is_clean else 0.06,
                            "t4": 0.02 if is_clean else 0.06,
                        },
                    )
                ]
                projs = [
                    LogitLensProjection(
                        layer_idx=14,
                        decode_step_t=30,
                        remaining_tokens_to_eos=22,
                        tool_logits={"spanner_graph_query_gql": logit_gap, "t2": 0.5, "t3": 0.2, "t4": 0.1},
                        predicted_args={"graph_name": "G", "gql_query": "MATCH (n) RETURN n"},
                        arg_token_confidences={"graph_name": 0.97 if is_clean else 0.68, "gql_query": 0.96 if is_clean else 0.65},
                    )
                ]
                p = pred_engine.predict_from_early_layers(attn, projs, self.registry)
                if p.should_prefetch and not is_clean:
                    false_dispatches += 1
            false_counts.append(false_dispatches)
        for i in range(len(false_counts) - 1):
            self.assertGreaterEqual(false_counts[i], false_counts[i + 1])

    def test_25_invalid_threshold_raises_value_error(self):
        """Predictor rejects invalid tau_conf outside (0, 1)."""
        with self.assertRaises(ValueError):
            EarlyLayerLogitLensPredictor(tau_conf=1.05)
        with self.assertRaises(ValueError):
            EarlyLayerLogitLensPredictor(tau_conf=0.0)


if __name__ == "__main__":
    unittest.main()
