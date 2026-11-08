+++
title = "The joint gluing inequality in the language of Green's functions"
date = "2026-11-08"
tags = ["percolation", "probability", "markov-chains"]
description = "A short expository note retelling OpenAI's proof of the joint gluing inequality with two Green functions: the hitting matrix is the Green matrix of a killed relay chain, and its positivity comes from one Poisson equation."
math = true
+++

The *joint gluing inequality* is the step that my [animated tour of the θ(p_c) = 0 proof](/post/theta-pc-visualized/) calls the hard part. I wrote a short note that retells OpenAI's proof of it in the language of Markov-chain potential theory: Green functions, killed chains, excessive functions and the Poisson equation. Probabilists who work with random walks already speak this language.

The note is an exposition. The estimates and key identities are OpenAI's, and it claims no new theorem. It adds only a change of language and a few elementary rearrangements, which it flags as such.

**Read the note (7 pages):** [Zenodo, doi:10.5281/zenodo.24062184](https://doi.org/10.5281/zenodo.24062184) · [PDF](/files/pei-2026-gluing-green-functions.pdf)

<!--more-->

## The inequality

Take independent percolation on a graph with countably many vertices and edges, with edge probabilities in \([0,1]\). Write \(C(x)\) for the open cluster of \(x\). For a vertex \(o\), a nonempty vertex set \(A\) and any vertex set \(T\),

\[
\mathbb P(o\leftrightarrow A,\ o\leftrightarrow T)\ \ge\ \mathbb P(o\leftrightarrow A)\,\inf_{a\in A}\mathbb P(a\leftrightarrow T).
\]

With \(T=\{b\}\), and after enlarging the event on the left, this is the multiplicative gluing inequality of [Kozma and Nitzan](https://arxiv.org/abs/2401.12397) (their Conjecture 1). They reduced \(\theta(p_c)=0\) on \(\mathbb Z^d\) to conjectured inequalities of this kind.

## The matrix H, on a 4-cycle

List the relays \(x_1,\dots,x_N\) and put \(o\) last. Let \(E_k\) be the event that the cluster of \(x_k\) avoids the earlier relays, and \(d_k=\mathbb P(E_k)\). Column \(k\) of the matrix \(H\) describes that conditioned cluster, and row \(i\) asks whether it contains \(x_i\):

\[
H_{ik}=\mathbb P\bigl(x_i\in C(x_k)\mid E_k\bigr).
\]

So \(H\) is unit lower triangular. The organizing fact is the *first-label partition*: the event that \(x_k\) is the first listed vertex of \(C(x_i)\) is \(E_k\cap\{x_i\in C(x_k)\}\), with probability \(H_{ik}d_k\). Each row therefore satisfies \(\sum_k H_{ik}d_k=1\).

<figure class="math-figure">
  <a href="/img/post/gluing-green/h-matrix.svg" target="_blank" rel="noopener">
    <img src="/img/post/gluing-green/h-matrix.svg" width="693" height="467" loading="lazy" decoding="async" alt="The 4-by-4 lower-triangular matrix H for a 4-cycle with vertices x1, x2, x3, o. Each column header is a small drawing of the cycle with the root vertex in blue and the earlier relays crossed out. The entries are exact: row o reads 9/16, 1/7, 1/3, 1, and the d row reads 1, 7/16, 3/8, 1/4. A callout explains that the entry in row o, column x2 is the probability that o is in the cluster of x2 given that x1 is not, which equals 1/7.">
  </a>
  <figcaption>The matrix \(H\) for the 4-cycle with relays listed \(x_1,x_2,x_3,o\) and every edge open with probability \(\tfrac12\), computed exactly. Column \(k\) is the cluster of \(x_k\) (blue), conditioned to avoid the earlier relays (crossed); row \(i\) asks whether it contains \(x_i\) (red ring). In row \(o\) the products \(H_{ok}d_k=\tfrac9{16},\tfrac1{16},\tfrac18,\tfrac14\) are the law of the first listed vertex of \(C(o)\), and they sum to 1.</figcaption>
</figure>

## H is a Green matrix

Put \(\Gamma=I-H^{-1}\), which is strictly lower triangular. The central lemma of OpenAI's proof is that \(\Gamma\ge0\) entrywise. Granting it, row \(i\) of \(\Gamma\) consists of nonnegative numbers on earlier relays, with \(\sum_j\Gamma_{ij}=1-d_i\). So \(\Gamma\) is the transition kernel of a Markov chain on the relays: from \(x_i\) it jumps to an earlier relay \(x_j\) with probability \(\Gamma_{ij}\), and it is killed with probability \(d_i\). Indices only decrease, so \(H=\sum_{n\ge0}\Gamma^n\) is a finite sum, and

\[
H_{ik}=\mathbb P_i(\text{the chain visits }x_k).
\]

In words, \(H\) is the Green (visit) matrix of this killed *relay chain*. The chain's killing-site law, \(\mathbb P_i(\text{killed at }x_k)=H_{ik}d_k\), coincides with the law of the first listed vertex of \(C(x_i)\). The note uses this only at the level of laws: it claims no pathwise coupling of the chain with the clusters.

<figure class="math-figure">
  <a href="/img/post/gluing-green/relay-chain.svg" target="_blank" rel="noopener">
    <img src="/img/post/gluing-green/relay-chain.svg" width="565" height="271" loading="lazy" decoding="async" alt="The relay chain on the states x1, x2, x3, o drawn in a row. Arrows point to earlier relays: x2 to x1 with probability 9/16, x3 to x2 with 3/7, x3 to x1 with 11/56, and, in blue, o to x3 with 1/3 and o to x1 with 5/12. Dashed arrows down to a cemetery state carry the killing probabilities 1, 7/16, 3/8 and 1/4.">
  </a>
  <figcaption>The relay chain \(\Gamma=I-H^{-1}\) of the same example. Solid arrows carry jump probabilities and dashed arrows killing probabilities. There is no arrow \(o\to x_2\), yet \(H_{o,x_2}=\tfrac13\cdot\tfrac37=\tfrac17\) is the chance of visiting \(x_2\). The blue arrows from \(o\) carry the relay weights, which sum to \(\tfrac34=\mathbb P(o\leftrightarrow A)\).</figcaption>
</figure>

**Joint gluing is one step of the chain.** For an increasing function \(F\ge0\) of the cluster, put \(y_i=\mathbb E\,F(C(x_i))\). The full positivity statement (OpenAI's cone lemma, below) gives \((I-\Gamma)y\ge0\), that is,

\[
y\ \ge\ \Gamma y .
\]

So \(y\) is *excessive* for the relay chain, and [Gouéré](https://arxiv.org/abs/2610.30063) states this row inequality in exactly this form (his Theorem 2). Now list the vertices of \(A\) first, put \(o\) last, and take \(F(U)=\mathbf 1\{U\cap A\ne\varnothing,\ U\cap T\ne\varnothing\}\). Then \(y_i=\mathbb P(x_i\leftrightarrow T)\) for \(x_i\in A\), the last entry \(y_o\) is the left side of the inequality, and \(\sum_i\Gamma_{oi}=1-d_o=\mathbb P(o\leftrightarrow A)\). Hence, on a finite graph (OpenAI reduce to this case),

\[
\mathbb P(o\leftrightarrow A,\ o\leftrightarrow T)=y_o\ \ge\ \sum_i\Gamma_{oi}\,y_i\ \ge\ \mathbb P(o\leftrightarrow A)\,\min_{a\in A}\mathbb P(a\leftrightarrow T).
\]

The relay weights of the video are exactly the first-step probabilities \(\Gamma_{oi}\) of the relay chain from \(o\).

## Why Γ ≥ 0: one Poisson equation

Fix \(k\), write \(s=x_k\), and let \(\mu_k\) be the law of \(C(s)\) given \(E_k\). OpenAI's proof runs the two-cluster Gibbs sampler \(P\) of [van den Berg, Häggström and Kahn](https://arxiv.org/abs/math/0408176). Given the cluster of \(s\), it resamples the cluster of the earlier relays in the rest of the graph. It then resamples a fresh cluster of \(s\) where that cluster is absent. The law \(\mu_k\) is stationary, and every step isolates \(s\) with probability at least some \(\delta>0\). Write \(v_k^F\) for the column built from \(F\) under \(\mu_k\), so that \(h_k=v_k^{1}\) is column \(k\) of \(H\). OpenAI's *column identity* reads

\[
v_k^F=v_k^{PF}+\sum_{l>k}\frac{d_l}{d_k}\,v_l^{R_F},
\]

where the deleted-set residual \(R_F\) is nonnegative and increasing when \(F\) is. OpenAI iterate this identity and pass to the limit in a closed cone.

The Green-function form sums the iteration first. By the Doeblin bound, the series

\[
\Psi=\sum_{j\ge0}\bigl(P^jF-\mu_k(F)\bigr)
\]

converges uniformly. It is increasing when \(F\) is, and it solves the *Poisson equation* \(\Psi-P\Psi=F-\mu_k(F)\): \(\Psi\) is the sampler's own Green function applied to the centred \(F\). The column identity is linear in \(F\), so one application to \(\Psi\) gives

\[
v_k^F-\mu_k(F)\,h_k=\sum_{l>k}\frac{d_l}{d_k}\,v_l^{R_\Psi}.
\]

By downward induction on \(k\), each \(v_l^{R_\Psi}\) lies in the cone spanned by the columns \(h_j\), \(j\ge l\). So the left side lies in the cone of the later columns. That is the cone lemma, and taking \(F\) to be "the cluster meets a later relay" gives \(\Gamma\ge0\).

The proof is no shorter. The Doeblin bound is still what makes the sum converge; only the closed-cone limit disappears.

## What is new, and what is not

- The estimates and identities are OpenAI's, from Section 6 of their preprint: the first-label partition, \(H^{-1}=I-\Gamma\) with \(\Gamma\ge0\), the column identity, the deleted-set residual and the Doeblin bound. The preprint has not been peer reviewed.
- The sampler and its iteration go back to van den Berg, Häggström and Kahn. Leder uses the same scheme in his guide to the formalized proof. Analogous iterations appear in the lecture notes prompted by A. Bou-Rabee, in [Blanc-Renaudie's notes](https://arxiv.org/abs/2610.10502) and in Gouéré's exposition.
- The Poisson equation and the fundamental matrix are classical (Kemeny and Snell).
- The note's own additions are elementary rearrangements: the Green-matrix reading of \(H\) at the level of laws, the Poisson form of the iteration, an explicit coefficient recursion, and the extensions to real-valued \(F\).

### Sources

- Yan Ru Pei, *The joint gluing inequality in the language of Green's functions: An exposition of a proof by OpenAI*, 2026, [doi:10.5281/zenodo.24062184](https://doi.org/10.5281/zenodo.24062184).
- OpenAI, *No percolation at criticality on quasi-transitive graphs*, preprint, 24 September 2026 ([openai/math](https://github.com/openai/math)), Section 6.
- G. Kozma, S. Nitzan, *A reduction of the θ(p_c) = 0 problem to a conjectured inequality*, [arXiv:2401.12397](https://arxiv.org/abs/2401.12397) (2024).
- J. van den Berg, O. Häggström, J. Kahn, *Some conditional correlation inequalities for percolation and related processes*, [arXiv:math/0408176](https://arxiv.org/abs/math/0408176) (2006).
- A. Blanc-Renaudie, *Notes on Kozma and Nitzan's proof of θ(p_c) = 0*, [arXiv:2610.10502](https://arxiv.org/abs/2610.10502) (2026).
- J.-B. Gouéré, *On the proof of the Kozma–Nitzan conjecture*, [arXiv:2610.30063](https://arxiv.org/abs/2610.30063) (2026).
- J. G. Kemeny, J. L. Snell, *Finite Markov Chains*, Van Nostrand (1960).
