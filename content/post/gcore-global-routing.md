+++
title = "Multi-DC KV-Aware Routing with Gcore"
date = "2026-07-20"
tags = ["llm-inference", "distributed-systems", "routing", "rust"]
+++

I worked with Gcore's AI team on the Dynamo side of a global inference-routing problem: when the same model is served in many data centers, how do you decide where each request should go without throwing away useful KV cache state?

<!--more-->

Inside one Dynamo deployment, the KV-aware router can choose a worker by looking at cached prefix overlap and active load. Across data centers, the same idea becomes harder. A useful decision has to balance three signals:

- **Cache locality** — which data center already holds the longest matching prefix.
- **Live load** — whether that data center has enough serving capacity right now.
- **Network latency** — whether the saved prefill work is worth a WAN hop.

The cache index is the uncomfortable part. Shipping an exact global radix tree from every site would be far too expensive at this scale. I co-developed a transposed Cuckoo-filter indexer and Relay design with Gcore engineer Nikita Sukharev. Each data center maintains exact local ownership, publishes a compact probabilistic projection, and lets a global consumer compare overlap across sites without centralizing the KV cache itself.

The work landed upstream in several pieces:

- [Multi-DC KV-aware routing design](https://github.com/ai-dynamo/dynamo/issues/11225)
- [Relay-shaped transposed Cuckoo-filter indexer](https://github.com/ai-dynamo/dynamo/pull/11435)
- [Sequenced DC Relay and domain-scoped global consumer](https://github.com/ai-dynamo/dynamo/pull/11793)
- [End-to-end Cuckoo-filter integration and documentation](https://github.com/ai-dynamo/dynamo/pull/11900)

Gcore built and measured its end-to-end global-router implementation over a real WAN. Its public evaluation covered 13 geographically distributed data centers, 4–239 ms RTT, and roughly 130 million KV-cache blocks. Gcore reported that a cold router replica resynchronized the whole grid in about 11 seconds. At concurrency 256, routing across the grid reached 114 requests/s versus 77 requests/s for one data center, while p95 latency fell from 1,129 ms to 872 ms.

Those are Gcore's system measurements; my upstream work focused on the Dynamo cache-index and Relay/consumer boundary. The multi-DC path is still evolving, particularly around non-local transport, recovery, and request forwarding, but the result is a concrete example of Dynamo's routing machinery being lifted from worker scale to data-center scale.

Read Gcore's full technical writeup:

- [Gcore introduces Global Inference Routing accelerated by NVIDIA Dynamo](https://gcore.com/blog/global-inference-routing)

[Back to projects](/post)
