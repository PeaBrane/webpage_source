+++
title = "NVIDIA Dynamo: Distributed LLM Inference"
date = "2025-10-01"
tags = ["llm-inference", "distributed-systems", "rust"]
+++

[Dynamo](https://github.com/ai-dynamo/dynamo) is NVIDIA's open-source datacenter-scale distributed inference serving framework for generative AI and reasoning models. Built in Rust for performance and Python for extensibility, it supports disaggregated prefill and decode, dynamic GPU scheduling, and LLM-aware request routing across multi-node multi-GPU topologies. The project has 6k+ GitHub stars and supports backends including TensorRT-LLM, vLLM, and SGLang.

<!--more-->

I worked on three core components:

- [**KV-Aware Router**](#kv-aware-router) — A hybrid router with two subsystems, the **Indexer** and the **Slot Tracker**, that together enable full KV-aware load balancing across workers.
- [**Engine Mockers**](#engine-mockers) — Lightweight engine simulators written entirely in Rust that replicate scheduling, block management, and timing behavior without requiring GPUs.
- [**Data Synthesizer**](#data-synthesizer) — A trace-driven workload generator that learns the statistical structure of a real request trace and synthesizes new datasets with tunable knobs.

---

## Table of Contents

1. [KV-Aware Router](#kv-aware-router)
    - [Indexer](#indexer)
    - [Slot Tracker](#slot-tracker)
    - [Event-Driven Updates](#event-driven-updates)
    - [Putting It Together: Full KV-Aware Load Balancing](#putting-it-together-full-kv-aware-load-balancing)
2. [Engine Mockers](#engine-mockers)
    - [Block Manager & Evictor](#block-manager--evictor)
    - [Scheduler](#scheduler)
    - [Performance Model](#performance-model)
    - [Scalability](#scalability)
3. [Data Synthesizer](#data-synthesizer)
    - [Extracting the Core Radix Tree](#extracting-the-core-radix-tree)
    - [Transition Probabilities & Sampling](#transition-probabilities--sampling)
    - [Tunable Knobs](#tunable-knobs)
    - [Open Questions](#open-questions)

---

## KV-Aware Router

The router is a **hybrid design** built from two complementary components: the **Indexer** and the **Slot Tracker**. The Indexer tracks *cached* KV blocks across the fleet (the global picture), while the Slot Tracker tracks *active* KV blocks on each engine (the local picture). Together they give the router enough information to proxy for both the prefill cost and the decode cost of routing a request to any given worker.

### Indexer

The Indexer maintains a **global overview of all cached KV blocks across all backend workers**. It is built on a [radix tree](https://en.wikipedia.org/wiki/Radix_tree), where each node in the tree corresponds to a block hash and is annotated with the set of workers that have that block cached. When a new request arrives, the router hashes its token sequence into block hashes and walks the radix tree to compute the **prefix overlap** (cache affinity) with each worker. This overlap directly tells us how many tokens the engine can skip during prefill — a longer overlap means a cheaper prefill.

The Indexer runs as a single-threaded [actor](https://en.wikipedia.org/wiki/Actor_model): one dedicated thread owns the radix tree and processes all events (store, remove, match) sequentially through a channel. This keeps the data structure simple and lock-free — no concurrent access, no synchronization overhead. A multithreaded variant (the *Flash Indexer*) that relaxes this constraint for higher throughput is in the works — more on that in a future post.

### Slot Tracker

The Slot Tracker (`ActiveSequences` / `ActiveSequencesMultiWorker`) tracks the **active blocks currently held by each engine** — i.e., the blocks that are in-flight for running requests. It maintains a per-worker view of which block hashes are alive (via reference-counted entries), how many prefill tokens are outstanding, and uses shared `Rc` pointers to deduplicate blocks that appear in multiple concurrent requests on the same worker.

Because active blocks are **ephemeral** (they exist only for the lifetime of a request), and because we can deterministically predict them from the request-response cycle (we know exactly when a request starts and ends), the Slot Tracker is updated **locally** by the router itself. There is no need to wait for the engine to tell us — we already know.

This local tracking also solves the **[thundering herd problem](https://en.wikipedia.org/wiki/Thundering_herd_problem)**: if multiple router frontends all query a shared remote state for active load, they may simultaneously see the same "lightly loaded" worker and all route to it. By tracking active slots locally, each router maintains its own consistent view and naturally avoids this kind of pile-on.

### Event-Driven Updates

The Indexer is updated by engines communicating **KV cache events** via a **[pub/sub pattern](https://en.wikipedia.org/wiki/Publish%E2%80%93subscribe_pattern)** (over [NATS](https://nats.io/)). This event-driven design is needed for two reasons:

1. **Eviction opacity** — The router cannot know when a cached block is evicted. Each engine has its own [eviction policy](https://en.wikipedia.org/wiki/Cache_replacement_policies) (LRU, LFU, etc.), and eviction happens asynchronously inside the engine. The engine *must* tell us, because there is no other way to know.

2. **Multi-frontend consistency** — In a deployment with multiple router replicas (frontends), the pub/sub fan-out ensures that all Indexers converge to the same global view. Every event is broadcast to all subscribers, so each frontend's radix tree stays in sync by virtue of receiving the same stream of events.

For the Slot Tracker, local tracking is sufficient since the router controls the request lifecycle. However, we also support optional **event-based syncing between routers** (via `ActiveSequenceEvent` published over [NATS](https://nats.io/)) for deployments that want cross-replica consistency on active load as well.

### Putting It Together: Full KV-Aware Load Balancing

The Indexer provides a proxy for the **prefill cost**: more cached prefix overlap with a worker means fewer new tokens to compute. The Slot Tracker provides a proxy for the **decode cost**: more active blocks on a worker means higher memory pressure and slower decode iterations (since decode is memory-bandwidth-bound). Together, these two signals allow the router to perform **full KV-aware load balancing** — routing each request to the worker that minimizes the total inference cost, not just naively round-robining or looking at queue depth.

---

## Engine Mockers

Mockers are **engine simulators** that do not require actual GPU engines to work. They replicate the core behavior of an LLM inference engine — block management, scheduling, eviction, and timing — entirely in Rust, allowing you to test and benchmark the serving infrastructure (router, orchestrator, autoscaler) in isolation.

### Block Manager & Evictor

The `KvManager` is a synchronous block manager that mirrors the block lifecycle of a real engine. It maintains two pools: **active blocks** (in-use by running requests, with reference counts) and **inactive blocks** (evictable, managed by an LRU evictor). It processes four types of `MoveBlock` signals:

- **Use** — Allocate or re-activate a block: if already active, increment its refcount; if inactive, promote it back to active; if absent, evict the LRU-oldest inactive block to make room.
- **Deref** — Decrement a block's reference count; when it hits zero, demote the block to the inactive pool.
- **Destroy** — Hard-remove a block from the active pool (e.g., on preemption).
- **Promote** — Convert a partial (generation-phase) block into a full block once it reaches block-size tokens.

The evictor itself (`LRUEvictor`) uses a `BTreeSet` keyed by insertion counter to maintain strict LRU ordering, matching the behavior of [vLLM](https://github.com/vllm-project/vllm)'s evictor.

The current "manual" backend tracks reference counts explicitly via `HashMap<UniqueBlock, usize>` — functional, but not the most idiomatic Rust. An in-progress integration with **KVBM** (the KV Block Manager library) replaces this with an [RAII](https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization)-based block lifecycle: blocks are reference-counted via smart handles, and deallocation happens automatically when the last handle is dropped. Beyond cleaner code, this also opens the door to **multi-tier memory simulation** (e.g., GPU HBM + CPU DRAM + NVMe), since KVBM natively models tiered storage with configurable eviction strategies (Lineage, LRU, Multi-LRU).

Crucially, the KvManager also **publishes KV cache events** (store / remove) to the same event sink that the real engines use. This means when a mocker evicts or allocates blocks, the Indexer in the router is updated exactly as it would be with a real engine — making the entire system testable end-to-end without GPUs.

### Scheduler

The `Scheduler` is an async scheduler that manages three queues — waiting, prefill, and decode — and drives the simulated forward pass loop:

1. **Receive** new requests and place them in the waiting queue.
2. **Schedule** waiting requests by checking resource budgets (watermark-based block budget, batched token budget, max sequence limit). If resources are insufficient, requests stay queued; if a running request must be preempted, the LRU-oldest decode request is evicted and re-enqueued.
3. **Simulate prefill** — compute the prefill duration and advance blocks through the KvManager. Chunked prefill is supported: if the token budget is partially consumed, only a chunk is prefilled per iteration.
4. **Simulate decode** — compute the decode duration based on active KV blocks, generate one output token per sequence, and check for completion or preemption.

### Performance Model

Timing simulation is driven by a `PerfModel` with two variants:

- **Polynomial (default)** — Hardcoded polynomial formulas: prefill TTFT scales quadratically with new tokens *(compute-bound)*, and decode ITL scales with active KV blocks *(memory-bandwidth-bound)*. This aligns with the same heuristics the router uses for cost estimation.

- **Interpolated** — Load your own profiler sweeps from an NPZ file. The model builds 1D interpolation for prefill (ISL → TTFT) and 2D [bilinear interpolation](https://en.wikipedia.org/wiki/Bilinear_interpolation) for decode (active KV tokens × context length → ITL). This lets you calibrate mockers to match real hardware profiles.

### Scalability

Since the mocker is fully in Rust with a thin Python LCI (Lightweight C Interface) wrapper, each engine instance is just a **tokio task** — not a process, not a GPU. You can spin up **1000+ simulated engines on a single node** (or more), making it practical to stress-test the router and orchestrator at datacenter-scale fleet sizes without any hardware.

---

## Data Synthesizer

The Data Synthesizer takes an existing **hash trace** (e.g., the Mooncake trace) and generates new synthetic datasets that preserve the statistical structure of the original while allowing controlled modifications via tunable knobs — in particular, **prefix length multiplier** and **prefix root multiplier**.

### Extracting the Core Radix Tree

The crux of the synthesizer is figuring out what the *prefixes* are in a trace. This is straightforward: if a block hash appears more than once across requests, it's part of a shared prefix. But we need more than just the set of prefixes — we need the **statistical properties of the underlying radix tree**.

The synthesizer builds a `networkx` DiGraph from the trace, where each node is a block hash and each edge records how many requests traversed it. It then:

1. **Verifies** the tree property (no node has more than one parent).
2. **Marks** leaf visits — nodes visited only once are unique user prompts, not shared context.
3. **Merges chains** — contracts unary paths (chains of nodes with one predecessor and one successor) into single nodes with a `length` attribute, converting the prefix tree into a compact **[radix tree](https://en.wikipedia.org/wiki/Radix_tree)**. This compression is important for efficient sampling.
4. **Removes leaves** — strips the unique-visit nodes, leaving only the **core radix tree** of shared prefixes. The removed leaf lengths are saved as a separate empirical distribution for later sampling.

### Transition Probabilities & Sampling

Each node in the core radix tree encodes its **transition probability to children nodes** via precomputed [CDFs](https://en.wikipedia.org/wiki/Cumulative_distribution_function) over edge weights. This means we can efficiently sample a path through the tree by walking from the super-root, sampling the next child at each node proportional to how frequently that transition appeared in the original trace, and stopping when we hit either an "end" sentinel (request terminates in the core tree) or a "cache end" sentinel (transition to a unique leaf path).

This assumes a kind of **[Markov property](https://en.wikipedia.org/wiki/Markov_property)** of request generation: the probability of visiting the next prefix block depends only on the current position in the tree, not the full history. For current workloads (multi-turn chat, document QA, few-shot prompting), this is a reasonable approximation.

Once the core path is sampled, the synthesizer appends a **unique leaf path** (user prompt) by sampling from the empirical distribution of leaf lengths extracted earlier. The input sequence length residual (tokens within the last partial block) and output sequence length are also sampled from their respective empirical distributions.

### Tunable Knobs

The synthesizer exposes several knobs for shaping the generated dataset, but the two most important are:

- **`prefix_len_multiplier`** — Scales the length of every node in the core radix tree. A multiplier of 2× doubles the number of shared context blocks per prefix, simulating workloads with longer system prompts or longer document contexts.
- **`prefix_root_multiplier`** — Replicates the entire core radix tree N times (with offset hash IDs). A multiplier of 4× means 4 independent prefix families, simulating a deployment serving 4× as many distinct applications or tenants.

Additional knobs include `prompt_len_multiplier` (scales unique user prompt lengths), `osl_multiplier` (scales output sequence lengths), and `speedup_ratio` (compresses inter-request arrival times).

### Open Questions

The [Markovian](https://en.wikipedia.org/wiki/Markov_property) assumption works well for today's workloads, but may need revisiting as more complex **agentic workflows** emerge — agent-subagent hierarchies, parallel tool calls, swarm architectures, etc. These patterns could introduce long-range dependencies in the prefix tree (e.g., an agent's prefix depends on the outcome of a subagent call three levels deep) that the current memoryless model wouldn't capture. There are many open statistical questions here around how to model the request-generation process for these more structured and dynamic workloads.

---

[Back to projects](/post)
