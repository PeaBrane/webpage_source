+++
title = "Frustration and Unsupervised Learning"
date = "2019-03-01"
tags = ["frustration", "rbm"]
+++

On a [signed graph](https://www.sciencedirect.com/science/article/pii/0166218X82900336) (or spin glass), often times there are interactions that cannot be satisfied regardless of how one assigns the spin states. This property is known as frustration. The study of [frustration of a bipartite graph](https://www.combinatorics.org/ojs/index.php/eljc/article/view/v19i4p10) is an active topic in graph theory, and has some important applications in unsupervised learning. For instance, a [highly frustrated RBM](https://jmlr.org/papers/v21/19-368.html) has several desirable statistical properties, such as the correspondence between the joint and marginal distribution, that can be leveraged to achieve more [efficient training](https://arxiv.org/abs/2001.05559).

<!--more-->

<p>
There is some numerical evidence that the complexity of a bipartite glass (or RBM) may display a double phase-transition as the frustration ratio is increased.</p>

![](/img/post/frus_comp.png) 

<p>
The frustration ratio may also be used as a proxy (in place of KL-divergence) in monitoring the training progress of the RBM, as it is expected to be more easily estimated.</p>

![](/img/post/frus_cd.png)

[Back to projects](/post)