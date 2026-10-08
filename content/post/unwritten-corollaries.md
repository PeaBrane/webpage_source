+++
title = "Corollaries Nobody Wrote Down"
date = "2026-10-08"
tags = ["mathematical-physics", "percolation", "probability", "disordered-systems", "quantum-spin-chains"]
description = "Four long-standing questions in mathematical physics that follow in one line from theorems in OpenAI's September 2026 manuscript collection, yet are stated nowhere: the 2D random ferromagnet, the spin-1 Heisenberg chain, and minimal spanning forests."
ShowToc = true
+++

*The machine didn't know what it had just proved.*

In September 2026 OpenAI posted a [collection of 719 mathematics manuscripts](https://github.com/openai/math/tree/fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb). Reading its mathematical-physics part, I kept running into the same pattern:
- a main theorem settles some statement *X*;
- the literature has known for decades that "if *X* then *Y*", where *Y* is itself a named open problem;
- nobody writes down *Y*. It isn't in the paper, and it isn't in a footnote either.

I searched all 719 manuscripts and the literature, and could not find any of the four corollaries below stated anywhere.

None of them needs new mathematics. Each one is an OpenAI theorem plus a classical result, and all the depth is in the cited papers. My guess is that this is an agentic-harness effect. The corollaries the manuscripts *do* state reuse the proof's own machinery (examples [below](#what-the-manuscripts-did-state)). What gets missed is noticing that a theorem answers someone else's question, posed in a different language. That takes a literature step nobody seems to have run.

<!--more-->

All links to OpenAI manuscripts point to commit [`fd4aeeb`](https://github.com/openai/math/tree/fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb) of `openai/math`. The manuscripts have not been peer reviewed, and each corollary inherits that status.

## 1. The 2D random ferromagnet has only two ground states

**What OpenAI proved.** [*No bigeodesics in planar first-passage percolation*](https://github.com/openai/math/blob/fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb/preprints/No-bigeodesics-in-planar-first-passage-percolation-September-24-2026/main.pdf) (result family 212). Put iid nonnegative random weights on the edges of ℤ². Assume their law has no atoms, and that the minimum of four independent copies has finite second moment. Then almost surely there is no *bigeodesic*: no doubly infinite self-avoiding path every finite piece of which is a shortest path between its endpoints.

**What was still open.** Take the disordered Ising ferromagnet on ℤ²:
- spins σ<sub>x</sub> = ±1;
- energy −Σ J<sub>xy</sub> σ<sub>x</sub>σ<sub>y</sub>, with iid random couplings J<sub>xy</sub> ≥ 0.

A *ground configuration* is one whose energy cannot be lowered by flipping finitely many spins. All-plus and all-minus always qualify. Are there any others?

[Bassan, Gilboa and Peled](https://arxiv.org/abs/2309.06437) call this "a long-standing challenge" (their Question 1.1). They show that non-constant ground configurations *do* exist in dimensions ≥ 4 for sufficiently concentrated disorder. In 2D the expected answer is no, but it had only been proved under unverified assumptions or for exactly solvable relatives of the model.

**The argument.** The key input is Newman's dictionary: non-constant ground configurations exist if and only if the dual first-passage model has bigeodesics. In that dual model, the weight of each dual edge is the coupling it crosses. The dictionary is in [*Topics in Disordered Systems*](https://doi.org/10.1007/978-3-0348-8912-4) (1997), Chapter 1, Propositions 1.1–1.2, and is sketched in [Damron–Hanson](https://arxiv.org/abs/1512.00804), §2.3. In words:

- **The wall is a path in a random metric.** Draw the *domain wall* of a configuration: the curve of dual edges crossing every bond whose two spins disagree. Its energy relative to all-plus is twice the total coupling it crosses, that is, twice its length in the random metric.
- **The wall must be locally shortest.** Flipping a finite block of spins reroutes the wall locally. So in a ground configuration no local reroute can shorten the wall, which means every finite piece of it is a shortest path.
- **No loops, so a bigeodesic appears.** With continuous couplings, shortest paths are unique, so the wall has no loops: a loop would give two different shortest paths between the same two points. A wall that never ends and has no loops contains a path that runs to infinity in both directions, and that is a bigeodesic.

OpenAI's theorem says no bigeodesic exists. So the wall is empty and the configuration is constant.

**Corollary.** Take iid couplings with an atomless law whose minimum of four copies has finite second moment, for example uniform or exponential couplings. Then almost surely the only ground configurations of the 2D disordered ferromagnet are all-plus and all-minus.

Amusingly, the OpenAI manuscript cites [Wehr and Woo](https://doi.org/10.1214/aop/1022855423). Their 1998 paper introduces the bigeodesic problem as being equivalent to exactly this ferromagnet question, yet the manuscript never mentions ferromagnets.

## 2. The spin-1 Heisenberg chain has a finite correlation length

**What OpenAI proved.** [*The periodic spin-one Haldane gap*](https://github.com/openai/math/blob/fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb/preprints/The-periodic-spin-one-Haldane-gap-September-24-2026/paper.pdf) (family 268). Take the antiferromagnetic spin-1 Heisenberg chain H = Σ **S**<sub>i</sub>·**S**<sub>i+1</sub> on a ring of L sites. For every even L ≥ 60 the ground state is unique and the spectral gap satisfies γ<sub>L</sub> > (4/105)·ln(80/79) ≈ 4.8×10⁻⁴.

**What was still open.** [Haldane's 1983 prediction](https://doi.org/10.1103/PhysRevLett.50.1153) for integer spin has two halves:
- a gapped, unique ground state;
- exponentially decaying correlations.

[Young's 2023 survey](https://arxiv.org/abs/2308.07848) says numerics support these properties for spin 1, "but a rigorous proof has still not been found". OpenAI's papers prove the first half. A [companion paper](https://github.com/openai/math/blob/fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb/preprints/A-boundary-field-gap-for-the-spin-one-Heisenberg-chain-September-24-2026/paper.pdf) also gets a topological index. Neither says anything about correlations.

**The argument.** It uses the *exponential clustering theorem* ([Hastings–Koma](https://arxiv.org/abs/math-ph/0507008); [Nachtergaele–Sims](https://arxiv.org/abs/math-ph/0506030); both 2006). Take a finite-range Hamiltonian with a gap above a unique ground state. Then connected correlations decay exponentially with distance, at a rate that depends only on the gap and the interaction strength.

The intuition:
- by the Lieb–Robinson bound, influence spreads at a bounded speed *v*;
- the gap makes the ground state forget a local disturbance within a time of about 1/γ;
- so two observables farther apart than roughly *v*/γ cannot stay correlated.

Every ring has the same nearest-neighbour terms, and OpenAI's gap holds uniformly in L. So the bound is uniform in L, and it passes to the infinite chain.

**Corollary.** In the ground state of the spin-1 Heisenberg ring, |⟨AB⟩ − ⟨A⟩⟨B⟩| decays exponentially in the distance between the local observables A and B, with constants independent of L. Every infinite-volume limit of these ground states therefore has a finite correlation length. That is the second half of Haldane's prediction.

The result is only qualitative. The proven gap is nearly a thousand times smaller than the true gap (about 0.41), so the correlation-length bound is far larger than the true value of about 6 lattice spacings. It also does not give area-law or matrix-product-state statements for open chains, which have spin-½ edge states.

## 3. Minimal spanning forest trees have one end, including on ℤ³

**What OpenAI proved.** [*No percolation at criticality on quasi-transitive graphs*](https://github.com/openai/math/blob/fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb/preprints/No-percolation-at-criticality-on-quasi-transitive-graphs-September-24-2026/paper.pdf) (family 213). On every infinite, connected, locally finite quasi-transitive graph with p<sub>c</sub> &lt; 1, critical Bernoulli bond percolation has no infinite cluster: θ(p<sub>c</sub>) = 0. This includes ℤ<sup>d</sup> for every d ≥ 2. On ℤ<sup>d</sup> this was previously known only for d = 2 and d ≥ 11 (I made a [video about the proof](/post/theta-pc-visualized/)).

**What was still open.** Give each edge an independent uniform label. The *wired minimal spanning forest* (WMSF) deletes every edge whose label is the largest on some cycle or on some bi-infinite path. Does every tree of this forest have exactly one end, that is, contain no doubly infinite path?

- [Alexander (1995)](https://doi.org/10.1214/aop/1176988378) proved this on ℤ<sup>d</sup> *assuming* θ(p<sub>c</sub>) = 0.
- [Lyons, Peres and Schramm (2006)](https://arxiv.org/abs/math/0412263), Theorem 1.1, proved it on every unimodular quasi-transitive graph under the same assumption.

So the conclusion was open exactly where θ(p<sub>c</sub>) = 0 was open, including ℤ³ through ℤ¹⁰.

**The argument.** Plug θ(p<sub>c</sub>) = 0 into those theorems. Roughly, Alexander showed that the only way a tree of the forest can have two ends is by swallowing an entire infinite cluster of critical percolation. With no infinite critical clusters, every tree has one end.

**Corollary.** On every unimodular quasi-transitive graph with p<sub>c</sub> &lt; 1 (in particular on ℤ³), almost surely every tree of the wired minimal spanning forest has one end. On amenable graphs such as ℤ<sup>d</sup> the free and wired forests coincide, so this is *the* minimal spanning forest. Alexander also notes a consequence for invasion percolation on ℤ<sup>d</sup>: from every site there is a unique optimal (minimax) path to infinity.

## 4. Free and wired minimal spanning forests differ on every nonamenable graph

**What OpenAI proved.** [*Nonuniqueness of percolation on nonamenable quasi-transitive graphs*](https://github.com/openai/math/blob/fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb/preprints/Nonuniqueness-of-percolation-on-nonamenable-quasi-transitive-graphs-September-24-2026/paper.pdf) (family 214) proves p<sub>c</sub> &lt; p<sub>u</sub> on every nonamenable quasi-transitive graph. This is the Benjamini–Schramm nonuniqueness conjecture: there is a window of p in which there are infinitely many infinite clusters.

**What was still open.** The *free* minimal spanning forest (FMSF) deletes only the edges whose label is the largest on some *finite* cycle. Lyons, Peres and Schramm asked, as [Question 6.6](https://arxiv.org/abs/math/0412263): does every nonamenable quasi-transitive graph satisfy FMSF ≠ WMSF? They also observed that a positive answer is equivalent to p<sub>c</sub> &lt; p<sub>u</sub>.

**The argument.** Their Proposition 3.6 says FMSF = WMSF exactly when, for almost every p, there is at most one infinite cluster. On quasi-transitive graphs, uniqueness monotonicity (Häggström–Peres, Schonmann) turns this into p<sub>c</sub> = p<sub>u</sub>.

In words, take an edge such that:
- its label p falls in the nonuniqueness window;
- its two endpoints lie in two *different* infinite clusters of the edges with lower labels.

No finite cycle through this edge has all its other labels below p, so the free forest keeps it. But a bi-infinite path running out through the two clusters makes it the largest label on that path, so the wired forest deletes it. When there are infinitely many infinite clusters, such edges exist with positive probability.

**Corollary.** On every nonamenable quasi-transitive graph, FMSF ≠ WMSF. This answers Lyons–Peres–Schramm's Question 6.6. Admittedly it is the cheapest of the four, since they had already reduced it to the Benjamini–Schramm conjecture.

## What the manuscripts did state

The manuscripts do derive corollaries, when the corollary reuses the proof's own machinery:

- The [SK limiting-law manuscript](https://github.com/openai/math/blob/fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb/preprints/The-low-temperature-Sherrington-Kirkpatrick-free-energy-limiting-law-September-24-2026/The-low-temperature-Sherrington-Kirkpatrick-free-energy-limiting-law-September-24-2026.pdf) derives disorder chaos from its variance asymptotics, via Chatterjee's identity.
- The [nonuniqueness manuscript](https://github.com/openai/math/blob/fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb/preprints/Nonuniqueness-of-percolation-on-nonamenable-quasi-transitive-graphs-September-24-2026/paper.pdf) derives the critical triangle condition and mean-field critical laws from its operator bound.
- The [boundary-field companion](https://github.com/openai/math/blob/fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb/preprints/A-boundary-field-gap-for-the-spin-one-Heisenberg-chain-September-24-2026/paper.pdf) derives Tasaki's topological index from its gap.

What gets missed is the other kind of corollary: noticing that the theorem answers a question posed elsewhere, in a different language (spin systems, spanning forests, correlation decay). That requires a step like "search the literature for 'if *X* then …' and 'equivalent to *X*'". A human specialist gets that step for free from years of talks. A pipeline that writes one focused manuscript per result apparently does not run it.

## Aside: why the bigeodesic trick needs a ferromagnet

The first corollary works because a ferromagnet has a frustration-free reference state. All-plus satisfies every bond, so a domain wall's cost is a positive, additive length. Walls are then geodesics in a random metric, and subadditivity, planar topology and decades of first-passage tools all apply.

In a spin glass, couplings of both signs leave frustrated plaquettes, and no configuration satisfies every bond. The cost along an interface between two ground states becomes a signed sum with no triangle inequality. The planar topology survives but the metric does not. So the bigeodesic theorem says nothing about whether the 2D Edwards–Anderson spin glass has incongruent ground states.

## Fine print

- All four corollaries inherit the status of the underlying preprints, which have not been refereed.
- "Not written anywhere" means three things:
  - it is not in any of the 719 manuscripts at the pinned commit (I searched all of them);
  - it is not in the collection's Lean documentation or overview;
  - it is not in any of the literature I could find.

  If one of these is already in print, please tell me and I will update the post with credit.
- All of the depth is in the cited work: OpenAI's theorems, and the classical results of Newman, Alexander, Lyons–Peres–Schramm, Hastings–Koma and Nachtergaele–Sims.
