+++
title = "Flash Indexer for Dynamo"
date = "2026-02-01"
tags = ["llm-inference", "distributed-systems", "rust"]
+++

I worked on the [Flash Indexer](https://docs.nvidia.com/dynamo/dev/blog/flash-indexer), a high-throughput global KV-cache indexer for [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo). The problem is to track which inference workers hold which KV blocks, and then answer routing queries fast enough that the indexer itself does not become the bottleneck.

<!--more-->

The design evolved through several iterations: a naive nested dictionary, a Rust actor, an inverted index, a radix tree, a concurrent radix tree, and finally a positional indexer with jump search. The key idea in the final version was to treat block position as a first-class key, which made random access possible and let us skip through long shared prefixes instead of traversing block-by-block.

The result was an indexer that sustains on the order of 100M+ combined read/write operations per second on real trace replays, to the point where networking, tokenization, and hashing start to dominate. NVIDIA published the full writeup here:

- [Flash Indexer: A Story of Inter-Galactic KV Routing](https://docs.nvidia.com/dynamo/dev/blog/flash-indexer)

This was one of the more satisfying systems projects I have worked on: a very concrete bottleneck, several failed-but-useful intermediate designs, and a final data-structure change that produced a large step-function improvement.

[Back to projects](/post)
