"""
Data Models for Agent-Synapse:
1. Early-Layer Attention Entropy & Logit-Lens Telemetry
2. Conformal Tool Call Prediction & Idempotency Classification
3. Speculative FastMCP Wire Envelopes & Headers
4. Copy-on-Write (CoW) On-Chip SRAM/HBM KV-Cache Page Descriptors
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class IdempotencyClass(str, Enum):
    """Safety classification for speculative tool execution."""
    READ_ONLY = "READ_ONLY"
    IDEMPOTENT_DRY_RUN = "IDEMPOTENT_DRY_RUN"
    MUTATING_BLOCKED = "MUTATING_BLOCKED"


class AcceleratorType(str, Enum):
    """Supported heterogeneous AI accelerator targets."""
    CLOUD_TPU_V6E = "CLOUD_TPU_V6E"
    NVIDIA_B200 = "NVIDIA_B200"
    CLOUD_TPU_V5P = "CLOUD_TPU_V5P"
    NVIDIA_H100 = "NVIDIA_H100"


@dataclass(frozen=True)
class AcceleratorProfile:
    """Hardware memory hierarchy and SRAM/HBM page-table timing characteristics."""
    accelerator: AcceleratorType
    sram_capacity_mb: float
    hbm_bandwidth_tbps: float
    page_block_tokens: int = 16
    cow_pointer_swap_ns: float = 3800.0  # ~3.8 microseconds hardware page descriptor swap
    inter_token_decode_ms: float = 11.5

    @classmethod
    def tpu_v6e(cls) -> "AcceleratorProfile":
        return cls(
            accelerator=AcceleratorType.CLOUD_TPU_V6E,
            sram_capacity_mb=128.0,  # VMEM + CMEM on-chip SRAM
            hbm_bandwidth_tbps=1.60,
            page_block_tokens=16,
            cow_pointer_swap_ns=3400.0,
            inter_token_decode_ms=10.8,
        )

    @classmethod
    def nvidia_b200(cls) -> "AcceleratorProfile":
        return cls(
            accelerator=AcceleratorType.NVIDIA_B200,
            sram_capacity_mb=256.0,  # L2 + Tensor Memory (TMEM)
            hbm_bandwidth_tbps=8.00,
            page_block_tokens=16,
            cow_pointer_swap_ns=4100.0,
            inter_token_decode_ms=9.6,
        )


@dataclass
class ToolSchemaSpec:
    """Registered FastMCP tool metadata with explicit idempotency & parameter schema."""
    tool_id: str
    service_target: str  # e.g., "SPANNER_GRAPH_GQL", "BIGQUERY_SQL", "VERTEX_SEARCH", "ERP_MUTATION"
    idempotency_class: IdempotencyClass
    required_args: Tuple[str, ...]
    optional_args: Tuple[str, ...] = ()
    supports_snapshot_dry_run: bool = False
    mean_io_latency_ms: float = 240.0
    anchor_tokens: Tuple[str, ...] = ()


@dataclass
class AttentionHeadTelemetry:
    """Cross-attention distribution over tool schema tokens at intermediate layer l in [8, 16]."""
    layer_idx: int
    head_idx: int
    decode_step_t: int
    tool_attention_weights: Dict[str, float]  # tool_id -> summed attention mass on schema span
    argument_span_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class LogitLensProjection:
    """Intermediate hidden-state unembedding projection z^(l) = W_U * RMSNorm(h_t^(l))."""
    layer_idx: int
    decode_step_t: int
    remaining_tokens_to_eos: int
    tool_logits: Dict[str, float]
    predicted_args: Dict[str, Any]
    arg_token_confidences: Dict[str, float]


@dataclass
class SpeculativePrediction:
    """Calibrated early-layer prediction produced before <tool_call> EOS emission."""
    prediction_id: str
    decode_step_t: int
    predicted_tool_id: str
    predicted_args: Dict[str, Any]
    attention_entropy: float
    logit_lens_posterior: float
    argument_stability_score: float
    conformal_confidence: float
    nonconformity_score: float
    threshold_tau: float
    estimated_lead_time_ms: float
    should_prefetch: bool
    rejection_reason: Optional[str] = None


@dataclass
class SpeculativeWireEnvelope:
    """FastMCP wire request dispatched asynchronously with speculative headers."""
    request_id: str
    tool_id: str
    canonical_args: Dict[str, Any]
    headers: Dict[str, str]
    checkpoint_hash: str
    dispatched_at_ns: int
    completed_at_ns: int
    network_io_latency_ms: float
    status: str  # "PREFETCHED_OK", "BLOCKED_NON_IDEMPOTENT", "DRY_RUN_SANDBOXED", "CANCELLED"
    response_payload: Optional[Dict[str, Any]] = None


@dataclass
class KVCheckpointDescriptor:
    """On-chip SRAM copy-on-write (CoW) KV-cache page table micro-checkpoint."""
    checkpoint_id: str
    checkpoint_hash: str
    branch_token_index: int
    base_page_ids: Tuple[int, ...]
    cow_shadow_page_ids: Tuple[int, ...]
    pinned_sram_bytes: int
    created_at_ns: int
    accelerator: AcceleratorType = AcceleratorType.CLOUD_TPU_V6E
    is_active: bool = True
    spliced_payload_tokens: int = 0


@dataclass
class ToolCallAST:
    """Normalized Abstract Syntax Tree representation of a predicted or emitted tool call."""
    tool_id: str
    canonical_args: Dict[str, Any]
    ast_sha256: str


@dataclass
class ASTEquivalenceResult:
    """Result of comparing predicted vs. actual emitted <tool_call> ASTs at EOS."""
    is_equivalent: bool
    predicted_ast_hash: str
    actual_ast_hash: str
    mismatch_fields: Tuple[str, ...] = ()
    action_taken: str = "SPLICED_ZERO_WAIT"  # or "COW_ROLLBACK"
    sram_operation_latency_us: float = 0.0


@dataclass
class SpeculativeExecutionMetrics:
    """End-to-end telemetry for a single agent tool-calling turn."""
    trace_id: str
    tool_id: str
    idempotency_class: IdempotencyClass
    conformal_confidence: float
    speculative_dispatched: bool
    ast_equivalent_hit: bool
    rollback_triggered: bool
    decode_lead_time_ms: float
    backend_io_latency_ms: float
    baseline_serial_latency_ms: float
    synapse_effective_latency_ms: float
    latency_saved_ms: float
    sram_checkpoint_rollback_us: float
    wire_headers_emitted: Dict[str, str]
