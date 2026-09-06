+++
title = "KV- and Token-Aware Routing in Ray Serve LLM"
date = "2026-08-25"
tags = ["llm-inference", "distributed-systems", "routing", "ray"]
+++

I coauthored an [Anyscale technical deep dive](https://www.anyscale.com/blog/llm-kv-token-aware-routing) on integrating NVIDIA Dynamo's KV-aware selection machinery into Ray Serve LLM. The collaboration is a useful example of a systems component becoming more valuable when it can be embedded outside the system where it started.

<!--more-->

KV cache overlap is important, but it is not the routing objective by itself. It tells you how much prefill work a worker can skip; it does not tell you how much uncached prefill or active decode work remains. A router that chases the largest cache hit can still pile new work onto an overloaded replica.

The better signal is **token load**:

- remaining uncached input tokens approximate new prefill work;
- active prefill tokens account for work already admitted;
- active decode state approximates ongoing memory-bandwidth pressure.

The architectural goal was to make that selection logic reusable without requiring Ray to adopt the rest of the Dynamo runtime. On the Dynamo side, I built the [runtime-free SelectionService](https://github.com/ai-dynamo/dynamo/pull/10641) and [replica synchronization path](https://github.com/ai-dynamo/dynamo/pull/10745). Anyscale engineer Jeffrey Wang then contributed the [in-process Python binding](https://github.com/ai-dynamo/dynamo/pull/10766), allowing Ray to own worker registration, request flow, and lifecycle updates while calling Dynamo's selection core directly.

The resulting boundary is deliberately modular:

- Ray keeps its native event, request, and data planes.
- Dynamo maintains the KV index and computes cache-overlap and load-aware scores.
- Ray Serve LLM uses those scores to select the target replica and reports request lifecycle updates back to the selector.

KV-cache- and token-aware routing was completed in [Ray 2.58.0](https://github.com/ray-project/ray/releases/tag/ray-2.58.0). The integration preserves Ray's orchestration model while reusing Dynamo's Rust routing core, rather than maintaining a second implementation of the same stateful scheduling logic.

Read the full article for the routing model, diagrams, configuration, and workload results:

- [Optimizing LLM Serving Efficiency: Moving Beyond KV Cache Reuse to Token-Load Awareness with Ray Serve LLM](https://www.anyscale.com/blog/llm-kv-token-aware-routing)

[Back to projects](/post)
