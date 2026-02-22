+++
title = "Peapods: Ising Monte Carlo in Rust"
date = "2026-02-21"
tags = ["monte-carlo", "spin-glass", "rust"]
+++

Simulating [spin glasses](https://en.wikipedia.org/wiki/Spin_glass) at scale has always been a pain — Python is too slow, and C++ is too painful. [Peapods](https://github.com/PeaBrane/peapods) is my attempt at a middle ground: a Monte Carlo package for Ising spin systems where the hot loops are written in Rust and everything else stays in Python, glued together with [PyO3](https://pyo3.rs). You just `pip install peapods` and go.

<!--more-->

The package supports the usual suspects — [Metropolis](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) and Gibbs single-spin flips, [Swendsen–Wang](https://en.wikipedia.org/wiki/Swendsen%E2%80%93Wang_algorithm) and [Wolff](https://en.wikipedia.org/wiki/Wolff_algorithm) cluster updates, and [parallel tempering](https://en.wikipedia.org/wiki/Parallel_tempering) — on arbitrary-dimensional hypercubic lattices with arbitrary coupling constants. Replicas run in parallel via [Rayon](https://github.com/rayon-rs/rayon), so you get multi-core scaling essentially for free.

The whole thing started because I needed a fast, flexible simulator for my [spin glass phase transition](../glass/) work and got tired of fighting with [Numba](https://numba.pydata.org/). Rust turned out to be a surprisingly pleasant language for this kind of thing.

[Back to projects](/post)
