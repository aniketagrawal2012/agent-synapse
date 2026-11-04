"""
Agent-Synapse: Attention-Guided Speculative Tool Prefetching and Microsecond
SRAM KV-Cache Checkpointing Across Heterogeneous AI Accelerators.

Author: Aniket Agrawal (aniketagrawal@google.com, Google Cloud AI)
"""

from agent_synapse.models import (
    AcceleratorProfile,
    AcceleratorType,
    ASTEquivalenceResult,
    AttentionHeadTelemetry,
    IdempotencyClass,
    KVCheckpointDescriptor,
    LogitLensProjection,
    SpeculativeExecutionMetrics,
    SpeculativePrediction,
    SpeculativeWireEnvelope,
    ToolCallAST,
    ToolSchemaSpec,
)
from agent_synapse.predictor import EarlyLayerLogitLensPredictor
from agent_synapse.speculative_wire import SpeculativeFastMCPWirePipeline
from agent_synapse.kv_checkpoint import (
    ASTEquivalenceSplicer,
    SRAMKVCacheCheckpointer,
)
from agent_synapse.engine import AgentSynapseEngine

__all__ = [
    "AcceleratorProfile",
    "AcceleratorType",
    "ASTEquivalenceResult",
    "AttentionHeadTelemetry",
    "IdempotencyClass",
    "KVCheckpointDescriptor",
    "LogitLensProjection",
    "SpeculativeExecutionMetrics",
    "SpeculativePrediction",
    "SpeculativeWireEnvelope",
    "ToolCallAST",
    "ToolSchemaSpec",
    "EarlyLayerLogitLensPredictor",
    "SpeculativeFastMCPWirePipeline",
    "ASTEquivalenceSplicer",
    "SRAMKVCacheCheckpointer",
    "AgentSynapseEngine",
]

__version__ = "1.0.0"
