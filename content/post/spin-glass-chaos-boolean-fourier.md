+++
title = "Spin-glass chaos with Boolean Fourier analysis"
date = "2026-10-10"
tags = ["spin-glass", "probability", "fourier-analysis"]
description = "An elementary proof of site-overlap disorder chaos for the fair ±J Edwards–Anderson model: gauge symmetry forces connecting Fourier supports, and the Boolean noise operator damps them."
math = true
+++

Change a small fraction of the bonds in a spin glass. How much of the spin configuration changes? For the fair ±J Edwards–Anderson model, there is a short answer using Fourier analysis on the Boolean cube. The decisive observation is geometric: every surviving Fourier term for a two-spin correlation must contain a path between those spins. The usual noise operator then suppresses it.

This is an exposition of a known result. [Chatterjee's 2023 paper](https://arxiv.org/abs/2301.04112) proved site-overlap disorder chaos at zero temperature for Gaussian couplings. [Chen, Kim and Sen](https://arxiv.org/abs/2404.09409v4), first posted in 2024 and revised in 2025, extended the result to general symmetric disorder and all temperatures. Here is the fair-bimodal, bond-resampling specialization of their argument, written directly in the language of Boolean Fourier analysis.

<!--more-->

## The model and the quantity to control

Take a nonempty finite simple graph \(G=(V,E)\), with \(N=|V|\), no external field, and no fixed spins. This includes a nearest-neighbor box with free boundaries or a periodic torus. Give each edge an independent fair sign \(x_e\in\{-1,+1\}\). Set the coupling magnitude to one; another common magnitude can be absorbed into the inverse temperature \(\beta\).

The Gibbs measure is

\[
\mu_{x,\beta}(\sigma)
=Z_{x,\beta}^{-1}
\exp\!\left(\beta\sum_{\{u,v\}\in E}x_{\{u,v\}}\sigma_u\sigma_v\right).
\]

Write \(\langle\cdot\rangle_x\) for its expectation. At \(\beta=\infty\), use the uniform measure on all ground states. This convention matters: bimodal disorder can have many ground states.

For \(0<\varepsilon\le1\), construct \(x^\varepsilon\) by independently replacing each bond with a fresh fair sign with probability \(\varepsilon\). Replacement actually changes the sign with probability \(\varepsilon/2\). Draw \(\sigma\) from \(\mu_{x,\beta}\) and \(\tau\) independently from \(\mu_{x^\varepsilon,\beta}\), conditional on the two disorders. Their site overlap is \(R=N^{-1}\sum_i\sigma_i\tau_i\).

The square removes the arbitrary global spin orientation. Let
\(f_{ij}(x)=\langle\sigma_i\sigma_j\rangle_x\). Conditional independence gives

\[
\mathbb E_\varepsilon\langle R^2\rangle
=\frac{1}{N^2}\sum_{i,j}
\mathbb E_\varepsilon[f_{ij}(x)f_{ij}(x^\varepsilon)].
\]

Here the bracket averages the two spin samples; \(\mathbb E_\varepsilon\) averages the original bonds and their resampling. The problem has become a noise-stability calculation for bounded functions of independent bits.

## Gauge symmetry forces a path

For \(S\subseteq E\), let \(\chi_S(x)=\prod_{e\in S}x_e\). These Walsh characters form an orthonormal basis under the uniform bond law, so

\[
\begin{aligned}
f_{ij}(x)&=\sum_{S\subseteq E}\widehat f_{ij}(S)\chi_S(x),\\
\widehat f_{ij}(S)&=\mathbb E[f_{ij}(x)\chi_S(x)].
\end{aligned}
\]

Assume \(i\ne j\). Flip spin \(v\) and all bond signs incident to \(v\). Every interaction is unchanged after this joint transformation. Therefore \(f_{ij}\) changes sign exactly when \(v=i\) or \(v=j\). Meanwhile, \(\chi_S\) changes sign exactly when the degree of \(v\) in the subgraph \(S\) is odd.

The transformed bonds still have the uniform law. Changing variables in the Fourier coefficient gives

\[
\widehat f_{ij}(S)
=(-1)^{\mathbf 1_{\{v=i\}}+\mathbf 1_{\{v=j\}}+\deg_S(v)}
\widehat f_{ij}(S).
\]

**A nonzero coefficient requires odd degree at \(i,j\) and even degree everywhere else.**

Every connected component of a finite graph has an even number of odd-degree vertices. Thus \(i\) and \(j\) must lie in the same component of \(S\). That component contains a path between them, and consequently

\[
\widehat f_{ij}(S)\ne0
\quad\Longrightarrow\quad |S|\ge d_G(i,j).
\]

<figure class="boolean-proof-figure">
  <a href="/img/post/boolean-chaos/fourier-support.svg" target="_blank" rel="noopener">
    <img src="/img/post/boolean-chaos/fourier-support.svg" width="720" height="640" loading="lazy" decoding="async" alt="Two possible Fourier supports. The first connects i to j and has even degree at every other vertex, with an optional attached cycle. The second leaves i and j in different components and therefore has zero Fourier coefficient.">
  </a>
  <figcaption>Colored edges are the Fourier support S, not satisfied bonds or a physical spin cluster. A connecting path is necessary; cycles are allowed. The parity condition permits a coefficient to survive, but does not guarantee that it is nonzero.</figcaption>
</figure>

That is the model-specific step. It works for Gibbs correlations at every temperature because the spin flip is a bijection of configurations, including the ground-state set.

## The noise operator finishes the estimate

Set \(q=1-\varepsilon\). A resampled bit has conditional mean \(q x_e\), and the resampling is independent across edges. Hence
\(\mathbb E[\chi_S(x^\varepsilon)\mid x]=q^{|S|}\chi_S(x)\).
Orthogonality now gives the standard noise-stability identity:

\[
\mathbb E_\varepsilon[f_{ij}(x)f_{ij}(x^\varepsilon)]
=\sum_S q^{|S|}\widehat f_{ij}(S)^2.
\]

For \(i\ne j\), all surviving sets have at least \(d_G(i,j)\) edges. Parseval and \(|f_{ij}|\le1\) therefore give

\[
\begin{aligned}
0&\le \mathbb E_\varepsilon[f_{ij}(x)f_{ij}(x^\varepsilon)]\\
&\le q^{d_G(i,j)}\sum_S\widehat f_{ij}(S)^2\\
&\le q^{d_G(i,j)}.
\end{aligned}
\]

For \(i=j\), \(f_{ii}=1\), so the same bound holds with distance zero. For distinct disconnected vertices the correlation is zero by flipping one whole component. Summing yields

\[
\boxed{\displaystyle
\mathbb E_\varepsilon\langle R^2\rangle
\le \frac{1}{N^2}\sum_{i,j}q^{d_G(i,j)}.}
\]

At \(\varepsilon=1\), read the diagonal contribution as one: the bound is \(1/N\). The spectral calculation is the familiar one from [O'Donnell's *Analysis of Boolean Functions*, §2.4](https://www.cs.cmu.edu/~odonnell/papers/Analysis-of-Boolean-Functions-by-Ryan-ODonnell.pdf). No influence estimate or hypercontractive inequality is needed.

## How small can the perturbation be?

For a nearest-neighbor box of side length \(L\) in \(d\) dimensions, \(N=L^d\). The distance sum is bounded by the corresponding sum on the full lattice:

\[
\sum_j q^{d_G(i,j)}
\le \left(\sum_{m\in\mathbb Z}q^{|m|}\right)^d
=\left(\frac{1+q}{1-q}\right)^d.
\]

The same bound holds on the torus by choosing a shortest coordinate representative for each vertex. Thus, for \(0<\varepsilon\le1\),

\[
\mathbb E_\varepsilon\langle R^2\rangle
\le \frac{1}{L^d}
\left(\frac{2-\varepsilon}{\varepsilon}\right)^d
\le \left(\frac{2}{\varepsilon L}\right)^d.
\]

In particular, \(\varepsilon_L\to0\) is allowed as long as \(\varepsilon_L L\to\infty\). For example, replacing bonds at rate \(L^{-1/2}\) makes the overlap tend to zero. Markov's inequality turns the second-moment estimate into convergence in probability, under the full joint disorder-and-spin law.

The bound is uniform in temperature. It also holds for independently sampled uniform ground states, either by the same gauge argument or by taking \(\beta\to\infty\) in finite volume. It does not assert the conclusion for every deterministic rule that selects one of many ground states.

## Why the broader paper uses Hermite analysis

Chen, Kim and Sen represent symmetric couplings as an odd function of a Gaussian variable. Fourier–Hermite analysis then treats several disorder laws, two kinds of perturbation, and hypergraph interactions within one framework. They explicitly connect bond resampling to Boolean noise sensitivity, citing [Garban and Steif](https://doi.org/10.1017/CBO9781139924160).

For fair ±J couplings and resampling, the Walsh basis exposes the argument with less machinery: **gauge symmetry forces a path, and noise damps every Fourier term carrying that path**. This is the elementary specialization of their Lemma 2.2, Proposition 2.4, and Theorem 1.3.

Finally, disorder chaos compares *different* bond realizations. It does not by itself establish a finite-temperature ordered phase for a single realization: the unperturbed overlap can already be small. Fixed boundary spins or an external field also change the gauge argument and require a separate treatment.

### Sources

- Sourav Chatterjee, [*Spin glass phase at zero temperature in the Edwards–Anderson model*](https://arxiv.org/abs/2301.04112).
- Wei-Kuo Chen, Heejune Kim and Arnab Sen, [*Disorder Chaos in Short-Range, Diluted, and Lévy Spin Glasses*, v4](https://arxiv.org/html/2404.09409v4), especially §§2.2–2.3 and §3.
- Ryan O'Donnell, [*Analysis of Boolean Functions*](https://www.cs.cmu.edu/~odonnell/papers/Analysis-of-Boolean-Functions-by-Ryan-ODonnell.pdf), §2.4.
