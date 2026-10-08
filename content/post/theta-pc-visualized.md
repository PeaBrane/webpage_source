+++
title = "No Percolation at Criticality: the θ(p_c) = 0 Proof, Visualized"
date = "2026-10-07"
tags = ["percolation", "probability", "math-visualization"]
description = "A 13-minute animated tour of a recent proof that critical Bernoulli percolation has no infinite cluster, with a close look at the joint gluing inequality."
youtube_id = "MIpnHoLF5vE"
video_duration = "PT13M6S"
video_upload_date = "2026-10-07"

[cover]
  image = "/img/post/theta-pc-thumbnail.png"
  alt = "θ(p_c) = 0, visualized: a glowing critical percolation cluster"
  hiddenInSingle = true
+++

Open each edge of a lattice independently with probability *p*. Above a critical value *p<sub>c</sub>* an infinite open cluster appears. But is there one exactly *at* *p<sub>c</sub>*? This is the θ(p<sub>c</sub>) = 0 problem of [percolation theory](https://en.wikipedia.org/wiki/Percolation_theory). It was settled in two dimensions by Harris and Kesten, and in high dimensions by the lace expansion, but stayed open for dimensions 3 to 10 for decades. A September 2026 preprint posted by [OpenAI](https://github.com/openai/math) claims a proof for every ℤ<sup>d</sup> and every quasi-transitive graph with *p<sub>c</sub>* &lt; 1. I made a 13-minute animated tour of how the argument works.

<!--more-->

{{< youtube id="MIpnHoLF5vE" title="No Percolation at Criticality: the θ(p_c) = 0 Proof, Visualized" >}}

The video first maps the whole argument, then zooms into its hardest step.

For a slower, illustrated explanation of the geometric construction, read [From a Seed to an Infinite Cluster](/post/percolation-seeds-to-infinity/). It follows the cubic-lattice argument through seeds, buffers, quarter-faces, contractions, and exploration, taking gluing as given.

**The strategy.** Suppose an infinite cluster existed at *p<sub>c</sub>*. Then certain local "moves" (open paths from a small ball to a nearby target) are likely at *p<sub>c</sub>*, and since each is witnessed by finitely many edges, they stay likely at some *p′* slightly below *p<sub>c</sub>*. Chaining these moves through a lattice of corridors with an adaptive exploration, in which every tested site fails with probability at most ε given everything revealed so far, a Peierls-type count produces an infinite cluster at *p′* &lt; *p<sub>c</sub>*. That contradicts the definition of *p<sub>c</sub>*.

**The hard part.** Each hand-off inside a corridor needs the *joint gluing inequality*, conjectured by [Kozma and Nitzan](https://arxiv.org/abs/2401.12397):

<p style="text-align:center">ℙ(o ↔ A, o ↔ T) ≥ ℙ(o ↔ A) · min<sub>a∈A</sub> ℙ(a ↔ T).</p>

The video shows why the naive "first relay reached" argument fails (on a 4-cycle it loses a factor 1/2), how the proof replaces it with non-negative relay weights read off a triangular matrix (*H*<sup>−1</sup> = *I* − Γ), and why Γ ≥ 0, using the exact-cluster Markov property and alternating cluster resampling.

The main source is a preprint and has not been peer reviewed; the video presents its argument.

### Chapters

- [0:00](https://www.youtube.com/watch?v=MIpnHoLF5vE&t=0s) Title
- [0:15](https://www.youtube.com/watch?v=MIpnHoLF5vE&t=15s) Part 1 · Is there an infinite cluster at the critical point?
- [1:45](https://www.youtube.com/watch?v=MIpnHoLF5vE&t=105s) Part 2 · The proof at a glance
- [2:31](https://www.youtube.com/watch?v=MIpnHoLF5vE&t=151s) Likely local moves, kept below *p<sub>c</sub>*
- [3:12](https://www.youtube.com/watch?v=MIpnHoLF5vE&t=192s) Corridors and an adaptive exploration
- [4:32](https://www.youtube.com/watch?v=MIpnHoLF5vE&t=272s) Hand-offs: why gluing is the hard part
- [5:47](https://www.youtube.com/watch?v=MIpnHoLF5vE&t=347s) Gluing 1/3 · The first-relay trap
- [8:07](https://www.youtube.com/watch?v=MIpnHoLF5vE&t=487s) Gluing 2/3 · Relay weights
- [10:15](https://www.youtube.com/watch?v=MIpnHoLF5vE&t=615s) Gluing 3/3 · Why the weights are non-negative
- [12:03](https://www.youtube.com/watch?v=MIpnHoLF5vE&t=723s) Recap and references

### Credits

- Main source: OpenAI, *No percolation at criticality on quasi-transitive graphs*, preprint, 24 September 2026 ([openai/math](https://github.com/openai/math)).
- The gluing problem and its reduction: G. Kozma, S. Nitzan, [arXiv:2401.12397](https://arxiv.org/abs/2401.12397) (2024).
- First-contact comparison: J. Leder (2026); expositions prompted by A. Bou-Rabee (2026).
- Adaptive blocks: G. R. Grimmett, J. M. Marstrand, *The supercritical phase of percolation is well behaved* (1990).
- Alternating cluster resampling: J. van den Berg, O. Häggström, J. Kahn, [arXiv:math/0408176](https://arxiv.org/abs/math/0408176) (2006).
- Also: Aizenman–Kesten–Newman (1987); T. Hutchcroft, [arXiv:1605.05301](https://arxiv.org/abs/1605.05301) (2016); R. Tessera, M. Tointon, [arXiv:1908.06044](https://arxiv.org/abs/1908.06044) (2021); S. Martineau, V. Tassion, [arXiv:1312.1946](https://arxiv.org/abs/1312.1946) (2017); R. Fitzner, R. van der Hofstad, [arXiv:1506.07977](https://arxiv.org/abs/1506.07977) (2017).
- Music: Debussy, *Danseuses de Delphes*, *Des pas sur la neige*, *Les sons et les parfums tournent dans l'air du soir*, *La fille aux cheveux de lin*; Scriabin, Prelude Op. 11 No. 15; Ravel, *Le Gibet*; Godowsky, *Alt Wien*.
- Animated with [Manim Community](https://www.manim.community/).

### Transcript

<details>
<summary>Show the full transcript</summary>

<p><strong>Part 1 · The question.</strong> Bond percolation: each edge of the lattice is open independently with probability p. Open edges are drawn; the largest open cluster is highlighted. As p grows, clusters merge. Beyond a critical value p_c (here 1/2) an infinite cluster appears, and in a finite window the largest cluster spans it. θ(p) is the probability that the origin belongs to an infinite open cluster. It is zero below p_c and positive above. The question is what happens exactly at p_c: does θ rise continuously from zero, or does it jump, so that an infinite cluster already exists at p_c? The proof rules out the jump: there is no infinite cluster at p_c, that is, θ(p_c) = 0. Known before: d = 2 (Harris 1960, Kesten 1980) and d ≥ 11 (Hara–Slade 1990, Fitzner–van der Hofstad 2017). The cases 3 ≤ d ≤ 10 stayed open for decades. The new preprint claims every quasi-transitive graph with p_c &lt; 1, which is Benjamini and Schramm's 1996 conjecture, and hence every ℤᵈ with d ≥ 2.</p>
<p><strong>Part 2 · The proof at a glance.</strong> Suppose, for contradiction, that an infinite cluster exists at p_c. The argument splits by how fast balls grow in the graph. Exponential growth was settled by Hutchcroft (2016), and an intermediate case has its own two-cluster argument. ℤᵈ has polynomial growth: that is the case we follow. There the strategy is: find likely local moves at p_c, which by continuity stay likely slightly below p_c; then chain them through corridors into an infinite cluster at some p′ &lt; p_c. That contradicts the definition of p_c. Let's watch the strategy in motion. If an infinite cluster exists at p_c, it is unique and reaches out in every direction, so a fixed small ball touches it with high probability. From that ball, open paths then reach a small box one step ahead and up, and one step ahead and down, each with probability at least 1 − η (after rescaling the coordinates). Each move is witnessed by finitely many edges, so its probability is a polynomial in p, hence continuous. It therefore stays above 1 − η at some p′ slightly below p_c. Chain the moves: sites of an oriented lattice are joined by corridors, regions of the original graph. Two corridors meet only inside the small rectangle at a shared endpoint. Test sites layer by layer. Testing a site reveals only its incoming corridor; its two outgoing corridors are still untouched. So, whatever has been revealed so far, a tested site fails with probability at most ε. Each time a site passes, the probability that its fresh corridors will carry the cluster onward is computed and stored; that stored prediction is what makes the bound hold at the next test. (Illustration: failures drawn at random.) If only finitely many sites were good, the boundary of the good set would be a closed trail of some length l passing at least l/4 failed sites, and there are at most 4ˡ such trails. For small ε the total is below 1. So with positive probability infinitely many sites are good, each reached by an actual open path from the seed: an infinite open cluster at p′ &lt; p_c. Contradiction, so θ(p_c) = 0. Inside a corridor, the cluster is handed from one relay region to the next, all the way to the child site. Each hand-off must fail with probability at most κ, whatever happened before; summing over the hand-offs controls the whole corridor. The catch: the cluster arrives at a point of Bᵢ chosen by the very edges that must carry it onward. Making the hand-off bound uniform is the delicate part. That is the whole strategy. One ingredient is far more delicate than the rest: the hand-off bound, called joint gluing. Its proof is the most involved step. Let's zoom in on it. Joint gluing says: reaching the relays and the target is at least as likely as reaching the relays, times the worst relay's chance of reaching the target. We take it in three steps.</p>
<p><strong>Joint gluing, step 1 · Why the hand-off is subtle.</strong> One hand-off, abstractly: the cluster of o has reached a set A of relays, and each relay, on its own, reaches the target T with probability at least 1 − λ, close to 1. Must o then reach T? The catch: which relay o arrives at is decided by the same random edges that must carry it onward. Conditioning on the arrival can spoil the continuation. The smallest example is a 4-cycle with relays a₁ and a₂ and target b. Each relay reaches b either directly or around the cycle, so this probability tends to 1 as p → 1. Split according to the first relay reached, taking a₁ before a₂. Suppose a₂ comes first: o reaches a₂ but not a₁. That forces o–a₁ closed and o–a₂ open. It also forbids a₂–b and b–a₁ from both being open, since then o would reach a₁ through b. Three cases remain. The conditional chance that a₂ reaches b is the green share: p(1 − p) / (1 − p²) = p/(1 + p). Now let p → 1. It tends to 1/2, although each relay alone reaches b with probability tending to 1. Being reached first rules out the likeliest configuration, both edges at b open, and leaves two equally likely cases, only one of which connects. So a hand-off cannot be argued relay by relay through 'the first relay reached': that conditioning can cost a factor 1/2. A different decomposition is needed.</p>
<p><strong>Joint gluing, step 2 · The inequality and its relay weights.</strong> Joint gluing holds for independent edges with any probabilities, including 0 and 1, so it still applies after conditioning on edges already revealed. Subtracting it from ℙ(o ↔ A) gives the hand-off bound: if every relay reaches the target with probability at least 1 − λ, then reaching the relays but missing the target has probability at most λ, however many relays there are. Proof idea: find relay weights αₐ ≥ 0 that add up to ℙ(o ↔ A) and multiply the plain chances ℙ(a ↔ T), with no conditioning on which relay came first. Bounding the sum by its smallest term gives the theorem. List the relays x₁, x₂, … and put o last. Column k of the matrix H describes the cluster of xₖ, conditioned to avoid the earlier ones; entry (i, k) is the chance that it contains xᵢ. So H is unit lower triangular. Key lemma: H⁻¹ = I − Γ where no entry of Γ is negative. The relay weights are the last row of Γ, and they add up to ℙ(o ↔ A). Applying the same positivity to the vector of target chances gives the weighted bound. On this 4-cycle the weights are 0.516 and 0.474, nothing like the first-relay masses 0.973 and 0.017. They are not probabilities of any 'first' event, so the trap of step 1 does not apply. The bound checks out exactly: 0.963 ≤ 0.964. Everything rests on one fact: no entry of Γ is negative. Step 3 shows why.</p>
<p><strong>Joint gluing, step 3 · Why the weights are non-negative.</strong> Ingredient 1, fresh territory. To know the cluster of S exactly, you only look at edges touching it: open ones inside, closed ones on its boundary (dashed). Every other edge is untouched: given the cluster, the rest of the graph is still independent percolation with the original probabilities. Ingredient 2, alternating resampling. Given the cluster U of s, regrow the cluster W of S with U removed; then, given W, regrow the cluster of s with W removed. Repeat. The law of the cluster of s, conditioned to avoid S, is stationary for this two-step chain (van den Berg–Häggström–Kahn). One round splits the column for an increasing function F into the column for its one-round average PF, plus leftover terms carried by later relays. The leftovers are non-negative: removing more vertices can only shrink the cluster of s. Ingredient 3, forgetting. With probability at least a, every edge at s is closed and s is isolated, wherever the chain started. So repeated averaging flattens F to its mean, geometrically fast. Iterating, the column for F minus its mean is a non-negative mix of later columns. With F = 'the cluster meets a later relay', this is exactly H⁻¹ = I − Γ with Γ ≥ 0.</p>
<p><strong>Recap.</strong> If an infinite cluster existed at p_c, likely local moves would persist slightly below p_c, and corridors with an adaptive exploration would chain them into an infinite cluster below p_c. Every hand-off along the way is controlled by joint gluing. The result is a contradiction, so θ(p_c) = 0. The gluing inequality itself uses only the independence of the edges: no symmetry and nothing specific to the lattice. Its non-negative relay weights replace the first-relay decomposition that fails on a 4-cycle.</p>

</details>

[Back to projects](/post)
