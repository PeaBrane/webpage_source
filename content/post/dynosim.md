+++
title = "DynoSim: Simulating the Pareto Frontier"
date = "2026-05-29"
tags = ["llm-inference", "distributed-systems", "simulation", "optimization"]
+++

I worked on [DynoSim](https://developer.nvidia.com/blog/dynosim-simulating-the-pareto-frontier/) and coauthored the NVIDIA technical blog introducing it. DynoSim is a workload-driven discrete-event simulation of the [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) serving stack: a digital twin that combines engine timing, scheduler behavior, routing, planning, KV-cache effects, and workload traces on one virtual timeline.

<!--more-->

## A Digital Twin of the Serving Stack

LLM serving behavior emerges from a stack of interacting decisions. The engine scheduler chooses batches; the router changes where load and reusable KV state accumulate; the Planner changes capacity and topology; KV movement changes what future requests can reuse. A timing estimate for an isolated forward pass cannot capture those feedback loops by itself.

[![DynoSim digital-twin architecture for the Dynamo serving stack](/img/post/dynosim-architecture.png)](/img/post/dynosim-architecture.png)

DynoSim composes the same conceptual pieces as a deployment—dynamic traffic, frontend lifecycle, KV-aware routing, aggregated or disaggregated workers, KVBM, and Planner decisions—around one event queue and one shared virtual clock. That common clock is the important part: every component sees the effects of earlier decisions before making its next one.

## Putting Timing Inside the Causal Loop

At the single-engine level, DynoSim separates two questions:

1. **Who runs next?** The backend scheduler forms a batch from the waiting and running requests using token budgets, sequence limits, KV capacity, cache state, and backend-specific policy.
2. **When does it finish?** A forward-pass model—optionally backed by [AIConfigurator](https://github.com/ai-dynamo/aiconfigurator)—maps that exact batch signature to a duration.

[![DynoSim scheduler and forward-pass timing loop](/img/post/dynosim-scheduler-loop.png)](/img/post/dynosim-scheduler-loop.png)

The completion becomes a future event. When the virtual clock reaches it, DynoSim updates emitted tokens, running requests, KV visibility, and router state; only then does the scheduler form the next batch. This preserves causality instead of estimating the time of a workload after its scheduling decisions have already been fixed.

The offline replay path is implemented in Rust and reuses the real Mocker scheduler cores for vLLM and SGLang. Multi-worker runs advance directly between request arrivals, forward-pass completions, prefill-to-decode handoffs, and worker-ready events without sleeping. The router can be round-robin or an in-process KV-aware router with its own radix-tree index and active-sequence state.

The workload can be a fixed request trace, synthetic traffic, or a multi-turn agent trace. For agentic workloads, replay preserves session and parent-child dependencies: changing a simulated model completion time moves dependent turns on the shared clock, while captured tool waits remain intact. It replays the serving workload rather than rerunning the agent's semantic decisions.

## Two Simulation Loops

[![DynoSim offline replay and Live Mocker simulation loops](/img/post/dynosim-simulation-loops.png)](/img/post/dynosim-simulation-loops.png)

DynoSim exposes two useful fidelity boundaries:

- **Offline replay** keeps the engine, router, Planner, KV, and workload models in virtual time. It is deterministic and fast, which makes it the right inner loop for algorithm comparisons and configuration sweeps.
- **Live Mocker** replaces the GPU engine but keeps the real Dynamo frontend, router, Planner, registration, request transport, and KV-event paths. It is slower, but it exposes distributed integration behavior and component overheads that an in-process replay intentionally removes.

Both consume the same workload contract. That lets a study move from a broad offline search to a smaller live-system test and finally to focused GPU validation without changing the question being asked.

## Sweeping the System

The simulation is faithful at the level of individual forward passes but fast enough to turn full-stack serving into an optimization loop. On an Apple M4 MacBook Air, the single-threaded Rust replay simulated the complete 23,608-request Mooncake trace with eight round-robin workers in 2.41 seconds. The modeled serving window was 60.1 minutes, so the run finished about 1,500× faster than real time.

I used that speed to run a full-grid Pareto campaign for MiniMax-M2.7 FP8 on B200 with vLLM 0.19.0. The sweep completed 8,688 deployment simulations spanning aggregated and disaggregated serving, worker splits, concurrency levels, and KV-router settings.

[![DynoSim full-grid capacity, latency, and interactivity Pareto frontiers](/img/post/dynosim-pareto-frontier.png)](/img/post/dynosim-pareto-frontier.png)

Each point is one full-trace simulation. Circles are aggregated deployments, triangles are disaggregated prefill/decode deployments, and color encodes the simulated GPU count. The left panel trades total output throughput against mean TTFT, so the interesting direction is up and left. The right panel trades total throughput against per-user output rate, so the interesting direction is up and right. The red curves are the exact non-dominated frontiers.

This is the payoff of simulating the system rather than just the engine. A locally attractive choice—more workers, a different prefill/decode split, stronger cache affinity—can move a bottleneck into queueing, decode pressure, or user interactivity. The Pareto frontier makes those system-level tradeoffs visible.

The goal is not to replace real-cluster validation. It is to make that validation focused: screen thousands of layouts and policies in simulation, carry only the Pareto candidates to real hardware, and use measured telemetry to recalibrate the next sweep.

- [DynoSim: Simulating the Pareto Frontier](https://developer.nvidia.com/blog/dynosim-simulating-the-pareto-frontier/)

[Back to projects](/post)
