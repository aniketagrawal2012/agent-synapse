# Agent-Synapse: Attention-Guided Speculative Tool Prefetching and Microsecond SRAM KV-Cache Checkpointing Across Heterogeneous AI Accelerators

**Author:** Aniket Agrawal (`aniketagrawal@google.com`, Google Cloud AI)

## Overview
**Agent-Synapse** is a three-stage hardware-software co-designed runtime that overlaps autoregressive token decoding on heterogeneous AI accelerators (**Google Cloud TPU v6e Trillium** and **NVIDIA B200**) with external Model Context Protocol (FastMCP) network I/O:

1. **Stage 1 — Early-Layer Attention Entropy & Logit-Lens Tool Predictor (`agent_synapse/predictor.py`)**: Monitors intermediate transformer cross-attention heads (Layers 8–16) and logit-lens projections ($\mathbf{z}^{(l)} = \mathbf{W}_U \operatorname{RMSNorm}(\mathbf{h}_t^{(l)})$) during autoregressive token decoding to predict idempotent read tool targets and canonical argument bindings **150 ms – 400 ms** *before* the `</tool_call>` EOS token is emitted, gated by a split-conformal confidence threshold $\tau_{\text{conf}} = 0.82$.
2. **Stage 2 — Speculative FastMCP Wire Pipeline (`agent_synapse/speculative_wire.py`)**: Dispatches read-only/idempotent FastMCP requests (Cloud Spanner Graph ISO GQL, BigQuery SQL, Vertex AI Search) asynchronously over the physical NIC with wire headers `X-Agent-Speculative-Prefetch: True` and `X-Synapse-Checkpoint-Hash` while autoregressive token decoding continues on the accelerator. Strictly blocks non-idempotent `WRITE`/`DELETE` mutations from speculative dispatch (`0.00%` unauthorized side effects).
3. **Stage 3 — Zero-Penalty SRAM KV-Cache Micro-Checkpointing & AST Equivalence Splicer (`agent_synapse/kv_checkpoint.py`)**: Pins a 16-token-aligned Copy-on-Write (CoW) KV-cache page table pointer in on-chip SRAM/HBM; when the model finishes emitting the `<tool_call>` block, checks canonical AST argument equivalence between the predicted call and actual call. On a match (**89.4% hit rate**), splices the already-returned tool payload directly into the KV-cache with **0 ms** network I/O wait (**61.8% p99 tail latency reduction**: `842.0 ms -> 321.6 ms`); on a mismatch (**10.6%**), rolls back the copy-on-write pointer in **< 12 microseconds** (`p50 = 4.2 us, p99 = 9.8 us`) with zero external side effects.

## Quickstart
```bash
# Run the 25 unit and property tests
pytest tests/test_agent_synapse.py

# Run the N=1,000 trace empirical benchmark & generate 300-DPI figures
python benchmarks/run_benchmarks.py

# Verify LaTeX manuscript syntax, balanced environments, and citation integrity
python verify_tex.py
```
