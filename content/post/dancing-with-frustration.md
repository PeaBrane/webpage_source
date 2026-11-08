+++
title = "Dancing with frustration: a review of short-range Ising spin glasses"
date = "2026-11-08"
tags = ["spin-glass", "mathematical-physics", "probability", "disordered-systems"]
description = "A 257-page review of the Edwards–Anderson model, as of November 2026: what is proved, what is believed, and where proofs break."
+++

I have posted a long review of the Edwards–Anderson spin glass: *Dancing with frustration. Short-range Ising spin glasses: what is proved, what is believed, and where proofs break.* It records where the mathematics of the short-range model stands as of November 2026.

**Read it (257 pages):** [Zenodo, doi:10.5281/zenodo.24061128](https://doi.org/10.5281/zenodo.24061128) · [PDF](/files/pei-2026-dancing-with-frustration.pdf) · certificate programs: [doi:10.5281/zenodo.24061127](https://doi.org/10.5281/zenodo.24061127)

<!--more-->

## The question

Take Ising spins on ℤ<sup>d</sup> and put an independent symmetric random coupling, for example ±J or Gaussian, on every nearest-neighbour edge. Then lower the temperature. Does the system freeze into a *spin-glass phase*, with long-range order in the overlap of two replicas? And if so, in which dimensions? This is the question Edwards and Anderson posed in 1975, when they introduced the model as the simplest short-range model of a spin glass.

Physicists have a fairly confident answer from numerics and renormalization-group reasoning: no transition at positive temperature in two dimensions, and a transition in three dimensions and above. Mathematically, almost none of this is proved. There is no proof of spin-glass order for symmetric independent couplings in any finite dimension, at any temperature. For undiluted laws such as ±J or Gaussian, there is also no proof that the two-dimensional model is disordered at all positive temperatures. For diluted laws just above the percolation threshold, that has been [proved](https://doi.org/10.5281/zenodo.24049737).

## Why another review

The standard accounts of spin-glass theory are mostly about mean field, centred on the Sherrington–Kirkpatrick model. The substantive mathematical treatments of the short-range model are Newman's lectures and the surveys and book of Newman and Stein, together with a recent chapter on metastates by Newman, Read and Stein. Much has happened since those accounts were written:

- zero-temperature chaos and droplet theorems;
- two-dimensional fluctuation bounds, and their limits;
- new graphical representations;
- classifications on hierarchical lattices;
- results on the Nishimori line;
- computer-assisted theorems.

The review is meant partly as a progress tracker.

I also expect the short-range spin glass to become one of the holy grails of mathematical physics in the coming years. It is a natural test bed for AI systems that do mathematics. The questions are precise and long open, and partial results come with checkable hypotheses. The barriers collected in the review give concrete test cases that any proposed argument must pass.

## Three threads

The subtitle names the three threads that run through the review.

- **What is proved.** These are the theorems about states, metastates, ground states, chaos, graphical representations and special lattices. Each is stated exactly and, where it teaches something, proved or sketched. Results from several programmes are restated in one notation.
- **What is believed.** These are the competing physical pictures and the numerics that calibrate them, kept visibly separate from the theorems. The pictures are replica symmetry breaking, droplets, trivial–nontrivial and chaotic pairs.
- **Where proofs break.** *Route warnings* record attractive lines of argument together with the lemma each one would need and the cases in which that lemma fails.

Every statement carries its status: theorem, conjecture, physics prediction, observation (with how it was checked), route warning or open problem. Results of my own programme are marked with their status. Each computer-assisted proof states what its certificate checks. A few results in the blue-percolation chapter come from a revised preprint that is not yet public, and they are marked as such where they appear.

## What is inside

1. **The question.** The model, gauge symmetry and what "order" means, together with the physical pictures and the numerics.
2. **States, metastates and two dimensions.** Metastates, the tool that makes "many states" a precise question, and the two-dimensional fluctuation programme.
3. **Zero temperature.** Ground states, domain walls and zero-temperature dynamics, then disorder chaos and superconcentration, including Chatterjee's zero-temperature theorems.
4. **Graphical representations.** The cluster representations proposed for frustrated systems, side by side, and two-replica blue percolation followed as far as it goes toward order.
5. **Toward non-order in low dimension.** High-temperature methods and the comparison with the random-field Ising model, whose analogous questions are solved. This part also develops a comparison with spanning forests and electrical networks.
6. **Testbeds and the Nishimori line.** Hierarchical and long-range proxies, diluted and decorated lattices, gauge symmetry, the Nishimori line and biased disorder.
7. **Where proofs break, and what to aim at.** The barriers that any proof of order or non-order must respect, small test cases for candidate inequalities, and open problems.

The reader's guide at the start suggests reading paths: for the open problems only, for techniques, for the cluster and comparison routes, and for readers coming from physics.

Related posts: [spin-glass chaos with Boolean Fourier analysis](/post/spin-glass-chaos-boolean-fourier/), an elementary exposition of one of the chaos results the review covers.
