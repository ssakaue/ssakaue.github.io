---
title: 'Online inverse linear optimization: Improved regret bound, robustness to suboptimality, and toward tight regret analysis'
authors:
- Shinsaku Sakaue
- Taira Tsuchiya
- Han Bao
- Taihei Oki
date: '2025-01-24'
publication_types:
- manuscript
publication: '*arXiv [cs.LG]*'

links:
#- name: Paper
#  url: 'https://openreview.net/forum?id=jHh804fZ5l&referrer=%5Bthe%20profile%20of%20Shinsaku%20Sakaue%5D(%2Fprofile%3Fid%3D~Shinsaku_Sakaue1)'
url_pdf: 'https://arxiv.org/abs/2501.14349'
url_code: ''
url_dataset: ''
url_poster: ''
url_project: ''
url_slides: ''
url_source: ''
url_video: ''
---

This post describes how to achieve the current best regret bound for online inverse linear optimization using the online Newton step (ONS), one of the main results in our recent paper. 
The method and analysis are very simple.

### Problem Setting

*Inverse optimization* is the problem of estimating parameters of forward optimization problems based on observed optimal solutions. 
Let's consider an *agent* who sequentially solves forward optimization of the following form $t = 1,\dots,T$:

$$
\mathrm{maximize}_{x \in \R^n} \quad {c^*}^\top x
\qquad
\mathrm{subject\ to} \quad x \in X_t,
$$

where $c^* \in \R^n$ is the agent's internal objective vector, $X_t \subseteq \R^n$ is the $t$-th feasible set (which may be non-convex). 
For simplicity, assume that $c^*$ and $X_1,\dots,X_t$ are contained in the unit ball, $\mathbb{B}^n$.
For each $t$, let $x_t \in \mathop{\mathrm{\arg\,\max}}_{x \in X_t} {c^*}^\top x$ be an optimal solution taken by the agent.

The goal of inverse optimization is to estimate the agent's objective vector $c^*$ based on the observed pairs of optimal solutions and feasible sets, $\{(x_t, X_t)\}_{t=1}^T$. 
Although this setting is elementary, it forms the basis for various problems of "inference from rational behavior," such as inferring customers' preferences from purchase history.


### Online-Learning Approach

Bärmann, Pokutta, and Schneider ([ICML 2017](https://proceedings.mlr.press/v70/barmann17a.html)) presented an elegant online-learning approach to this problem.
For each $t$, their method computes a prediction $\hat{c}_t$ of $c^*$ based on past observations, $\{(x_{i}, X_{i})\}_{i=1}^{t-1}$.
Let $\hat x_t \in \mathop{\mathrm{\arg\,\max}}_{x \in X_t} {\hat c_t}^\top x$ denote the solution suggested by the prediction $\hat{c}_t$.
A natural performance measure for those predictions is the *regret*: 

$$
R_T^{c^*} \coloneqq \sum_{t=1}^T {c^*}^\top(x_t - \hat x_t),
$$

which represents the cumulative gap between the optimal values and objective values achieved by following the predictions $\hat{c}_t$.
Bärmann et al. also defined another convenient measure that upper bounds the regret: 

$$
\tilde R_T^{c^*} \coloneqq \sum_{t=1}^T (\hat c_t - c^*)^\top (\hat x_t - x_t) = \sum_{t=1}^T \hat c_t^\top (\hat x_t - x_t) + R_T^{c^*} \ge R_T^{c^*},
$$

where the last inequality is due to $x_t, \hat x_t \in X_t$ and the optimality of $\hat x_t$ for $\hat c_t$.
This measure suggests how online learning can be applied.
Define $f_t\colon c\mapsto c^\top (\hat x_t - x_t)$ as the $t$-th cost function. 
Taking cost vectors, $\hat x_t - x_t$, to be chosen by an adversary, we can use standard online learning methods, such as the online gradient descent, to achieve 

$$
R_T^{c^*} \le \tilde R_T^{c^*} = \sum_{t=1}^T (f_t(\hat c_t) - f_t(c^*)) = O(\sqrt{T}).
$$

This ensures that the average regret vanishes as $T \to \infty$. 
Let us refer to the formulation of Bärmann et al.---sequentially computing $\hat{c}_t$ to bound $R_T^{c^*}$---as *online inverse linear optimization*. 

### Logarithmic Regret via Ellipsoid-Based Method
In general online linear optimization (OLO), the regret bound $O(\sqrt{T})$ is tight. So, is $R_T^{c^*} = O(\sqrt{T})$ also tight for online inverse linear optimization? 
The answer is no: Besbes, Fonseca, and Lobel ([COLT 2021](https://proceedings.mlr.press/v134/besbes21a.html), [Oper. Res. 2023](https://pubsonline.informs.org/doi/10.1287/opre.2021.0369)) achieved $R_T^{c^*} = O(n^4 \log T)$, where $n$ is the dimension of the space where $c^*$ lies. 

Their idea is novel and somewhat different from the online-learning approach, which I briefly describe here. 
Intuitively, the feedback $(x_t, X_t)$ is informative compared to that in general OLO; the optimality of $x_t \in X_t$ narrows down the possible existence of $c^*$ to the normal cone of $X_t$ at $x_t$. The method of Besbes et al. implements this idea by sequentially updating an ellipsoidal cone that is ensured to contain $c^*$, thereby inducing an appropriate amount of exploration. Based on the volume argument commonly used in the analysis of the ellipsoid method, they established $R_T^{c^*} = O(n^4 \log T)$. The runtime is polynomial in $n$ and $T$.

### Our Approach: Online Newton Step
We provide a simple and efficient method that achieves $R_T^{c^*} = O(n \log T)$, improving the previous logarithmic bound by a factor of $n^3$. 
Our approach is close to Bärmann et al., but we apply the online Newton step (ONS) to different cost functions $f_t$. 

ONS is an online convex optimization (OCO) method that achieves the logarithmic regret for exp-concave losses. 
We say a twice differentiable function $f\colon \R^n \to \R$ is *$\alpha$-exp-concave* on $\mathcal{K} \subseteq \R^n$ if $\nabla^2 f(x) \succeq \alpha \nabla f(x) \nabla f(x)^\top$ for all $x \in \mathcal{K}$. 

---
#### Regret bound of ONS ([Hazan et al. 2007, Theorem 2](https://link.springer.com/article/10.1007/s10994-007-5016-8)).
Let $\mathcal{K}$ be a convex set with diameter $D$ and $f_1,\dots,f_T\colon\R^n\to\R$ be $\alpha$-exp-concave loss functions. Assume $\| \nabla f_t(x) \| \le G$ for all $t$ and $x \in \mathcal{K}$. Let $\hat c_1,\dots,\hat c_T \in \mathcal{K}$ be the outputs of ONS. Then, for any $c^* \in \mathcal{K}$, it holds that  

$$\sum_{t=1}^T (f_t(\hat c_t) - f_t(c^*)) = O\left( n\left( \frac{1}{\alpha} + GD \right) \log T \right).$$

---

Our method simply applies ONS to appropriate exp-concave functions $f_t$.
Let $\eta = 1/8$ (the reason for this choice will be clear later) and define $f_t\colon \R^n \to \R$ by

$$
f_t(c) \coloneqq - \eta (\hat c_t - c)^\top (\hat x_t - x_t) + \eta^2 \left( (\hat c_t - c)^\top (\hat x_t - x_t) \right)^2,
$$

where $x_t$ is the agent's optimal solution, $\hat c_t \in \mathbb{B}^n$ is the $t$-th prediction, and $\hat x_t \in \mathop{\mathrm{\arg\,\max}}_{x \in X_t} {\hat c_t}^\top x$. 

Consider applying ONS to the above $f_t$, restricting the domain to $\mathbb{B}^n$. 
To derive a regret bound, we need to evaluate the parameters, $G$, $D$, and $\alpha$, in the ONS regret bound. 
In fact, these parameters are all constants. <details><summary>Here is the detailed calculation.</summary>

For simplicity, let $g_t = \hat x_t - x_t$, which satisfies $\| g_t \| \le 2$ since $\hat x_t, x_t \in X_t \subseteq \mathbb{B}^n$.

- Since the domain is $\mathbb{B}^n$, we have $D = 2$.
- By using $c, \hat c_t \in \mathbb{B}^n$, $\| g_t \| \le 2$, and $\eta = 1/8$, we have $$\| \nabla f_t(c) \| = \| \eta g_t - 2\eta^2 g_tg_t^\top (\hat c_t - c)\| \le 2\eta + 16\eta^2 \le 1/2$$ and $$\nabla f_t(c) \nabla f_t(c)^\top =  \eta^2 \left( 1 - 2 \eta g_t^\top (\hat c_t - c) \right)^2 g_t g_t^\top \preceq \eta^2 (1 + 8\eta)^2 g_t g_t^\top = 2 \nabla^2 f_t(c),$$ hence $G = \alpha = \frac{1}{2}$.

Thus, those parameters are constant for $\eta = 1/8$. 
</details>

Therefore, ONS applied to $f_t$ satisfies 

$$
\sum_{t=1}^T (f_t(\hat c_t) - f_t(c^*)) = O\left(n\log T \right).
$$

The remaining task is to bound $\tilde R_T^{c^*} = \sum_{t=1}^T (\hat c_t - c^*)^\top (\hat x_t - x_t)$, which upper bounds the regret $R_T^{c^*}$.
For convenience, define 

$$
V_T^{c^*} \coloneqq \sum_{t=1}^T \left( (\hat c_t - c^*)^\top (\hat x_t - x_t) \right)^2.
$$

Due to $\hat c_t, c^* \in \mathbb{B}^n$, $\hat x_t, x_t \in X_t \subseteq \mathbb{B}^n$, and $(\hat c_t - c^*)^\top (\hat x_t - x_t) \ge 0$ (by the optimality of $\hat x_t$ and $x_t$ for $\hat c_t$ and $c^*$, respectively), we have 

$$
V_T^{c^*} \le 4\sum_{t=1}^T (\hat c_t - c^*)^\top (\hat x_t - x_t) = 4\tilde R_T^{c^*}.
$$ 

By using this and $f_t(\hat c_t) = 0$, which follows from the definition of $f_t$, we have

$$
\tilde R_T^{c^*} = - \sum_{t=1}^T \frac{f_t(c^*)}{\eta} + \eta V_T^{c^*} \le \sum_{t=1}^T \frac{f_t(\hat c_t) - f_t(c^*)}{\eta} + 4\eta \tilde R_T^{c^*}.
$$

With $\eta = 1/8$, we obtain 

$$
\frac{\tilde R_T^{c^*}}{2}
\le 
8\sum_{t=1}^T (f_t(\hat c_t) - f_t(c^*)) = 
O(n\log T), 
$$

thus establishing $R_T^{c^*} \le \tilde R_T^{c^*} = O(n\log T)$.

### Related Topics

The above definition of $f_t$ and the proof strategy are inspired by those used in MetaGrad ([Van Erven and Koolen NeurIPS 2016](https://papers.nips.cc/paper_files/paper/2016/hash/14cfdb59b5bda1fc245aadae15b1984a-Abstract.html), [Van Erven et al. JMLR 2021](https://www.jmlr.org/papers/v22/20-1444.html)). Interestingly, using MetaGrad, instead of ONS, adds robustness against the suboptimality of the agent's solutions, which is also discussed in our paper. 
We have also obtained a lower bound of $R_T^{c^*} = \Omega(n)$ and an upper bound of $R_T^{c^*} = O(1)$ for the case of $n=2$; the algorithm for $n=2$ is a simple variant of Besbes et al. Closing the $\log T$ gap for general $n$ is an interesting open problem.

[Another paper](https://ssakaue.github.io/publication/sakaue-2025-revisiting/) of ours, which will appear in AISTATS 2025, also studies online inverse linear optimization. 
It views the problem as online convex optimization of a Fenchel–Young loss ([Blondel et al. JMLR 2020](https://jmlr.csail.mit.edu/papers/v21/19-021.html)) and presents a finite regret bound under the assumption that the agent's forward problems have a gap between the optimal and suboptimal objective values.