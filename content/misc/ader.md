# The Ader Algorithm  (Zhang et al. NeurIPS 2018)

I describe the basic idea of [Zhang et al., NeurIPS '18: "Adaptive online learning in dynamic environments,"](https://papers.nips.cc/paper_files/paper/2018/hash/10a5ab2db37feedfdeaab192ead4ac0e-Abstract.html) which I believe is a nice application of the multiple-learning-rate technique. The following analysis, which has an extra $\log\log T$, is looser than the original paper, but I hope it is easier to understand.


## Problem Setting
- $\mathcal{W} \subseteq \R^d$ is a closed convex domain.
- Bounded domain: $\|w - w'\| \leq 1$ for all $w, w' \in \mathcal{W}$ and $0 \in \mathcal{W}$.
- $f_t \colon \mathcal{W} \to \R$ is a convex loss function at round $t$.
- Bounded gradients: $\|\nabla f_t(w)\| \leq G$ for all $w \in \mathcal{W}$ and $t$.

### Online Learning Protocol
For $t = 1, 2, \ldots, T$:
1. Learner plays $w_t \in \mathcal{W}$ ($w_1$ is arbitrary).
2. Environment reveals $f_t$ (provides gradient access for every $w \in \mathcal{W}$).
3. Learner incurs loss $f_t(w_t)$ and computes $w_{t+1} \in \mathcal{W}$ based on observed information.

### Regret and Dynamic Regret
Let $u_1, \ldots, u_T \in \mathcal{W}$ be unknown comparators. 
In the static setting, $u_1 = \cdots = u_T = u$.

- Static Regret: $\sum_{t=1}^T \left(f_t(w_t) - f_t(u)\right)$

- Dynamic Regret: $\sum_{t=1}^T \left(f_t(w_t) - f_t(u_t)\right)$

## Online Gradient Descent (OGD)

- Learning rate: $\eta > 0$
- Update rule:
$$w_{t+1} = \arg\min_{w \in \mathcal{W}} \|w_t - \eta \nabla f_t(w_t) - w \|$$

## Static Regret Analysis
By the Pythagorean theorem, 

$$\|w_{t+1} - u\|^2 \leq \|w_t - \eta \nabla f_t(w_t) - u\|^2 = \|w_t - u\|^2 + \eta^2\|\nabla f_t(w_t)\|^2 - 2\eta \langle w_t - u, \nabla f_t(w_t) \rangle.$$

Since $\|\nabla f_t(w_t)\| \leq G$, 

$$\|w_{t+1} - u\|^2 \leq \|w_t - u\|^2 + \eta^2 G^2 - 2\eta \langle w_t - u, \nabla f_t(w_t) \rangle.$$

By telescoping and $\|w_1 - u\|^2 \leq 1$, 

$$\sum_{t=1}^T \langle w_t - u, \nabla f_t(w_t) \rangle \leq \frac{1}{2\eta}\left(\|w_1 - u\|^2 - \|w_{T+1} - u\|^2\right) + \frac{\eta G^2 T}{2} \leq \frac{1}{2\eta} + \frac{\eta G^2 T}{2}$$

From convexity, $f_t(w_t) - f_t(u) \leq \langle w_t - u, \nabla f_t(w_t) \rangle$, hence 

$$\sum_{t=1}^T f_t(w_t) - \sum_{t=1}^T f_t(u) \le \sum_{t=1}^T \langle w_t - u, \nabla f_t(w_t) \rangle \leq \frac{1}{2\eta} + \frac{\eta G^2 T}{2}.$$

Setting $\eta = \frac{1}{G\sqrt{T}}$, 

$$\sum_{t=1}^T f_t(w_t) - \sum_{t=1}^T f_t(u) \leq G\sqrt{T}.$$

## Dynamic Regret Analysis

- Comparators: $u_1, \ldots, u_T$.
- Path length: $P_T = \sum_{t=2}^T \|u_t - u_{t-1}\|$.

By similar analysis,

$$\sum_{t=1}^T f_t(w_t) - \sum_{t=1}^T f_t(u_t) \leq \sum_{t=1}^T \left(\frac{\|w_t - u_t\|^2 - \|w_{t+1} - u_t\|^2}{2\eta}\right) + \frac{\eta G^2 T}{2}.$$

The first sum on the r.h.s. is at most

$$\|w_1 - u_1\|^2 + \sum_{t=2}^T \left( \|w_t - u_t\|^2 - \|w_t - u_{t-1}\|^2 \right).$$

By the triangle inequality, the summand is at most

$$(\|w_t - u_t\| + \|w_t - u_{t-1}\|)(\|w_t - u_t\| - \|w_t - u_{t-1}\|) \le 2 \|u_t - u_{t-1}\|.$$

Therefore, we have 

$$\sum_{t=1}^T f_t(w_t) - \sum_{t=1}^T f_t(u_t) \leq \frac{1 + P_T}{\eta} + \frac{\eta G^2 T}{2}.$$

Setting $\eta \simeq \frac{1}{G \sqrt{T}}$,

$$\sum_{t=1}^T f_t(w_t) - \sum_{t=1}^T f_t(u_t) \lesssim G(1 + P_T)\sqrt{T}.$$

If $u_1 = \cdots = u_T = u$, then $P_T = 0$ and thus the $O(G\sqrt{T})$ static regret bound is recovered.

## Adapting to Unknown Path Length
If we knew $P_T$ in advance, we could set the learning rate to $\eta \simeq \frac{1}{G}\sqrt{\frac{1 + P_T}{T}}$ to achieve $O(G\sqrt{(1 + P_T)T})$, but we don't know $P_T$ in advance.


**Observation**: For any $\eta > 0$, OGD achieves 

$$\sum_{t=1}^T f_t(w_t) - \sum_{t=1}^T f_t(u_t) \lesssim \frac{1 + P_T}{\eta} + \eta G^2 T.$$

**Idea**: Try multiple learning rates and aggregate their outputs!

### Meta-learner and Experts

Ideal learning rate: $\eta^* = \frac{1}{G}\sqrt{\frac{1 + P_T}{T}}$ (unknown).

Let $\eta_{\min} = \frac{1}{G\sqrt{T}}$ and $\eta_{\max} = \frac{\sqrt{T}}{G}$, so that $\eta^* \in [\eta_{\min}, \eta_{\max}]$.

Define a set of learning rates:  

$$\mathcal{E} \coloneqq \left\{\eta_i = \eta_{\min} \cdot 2^{i-1} : i = 1, 2, \ldots, \lceil\log_2(\eta_{\max}/\eta_{\min})\rceil + 1\right\}.$$

**Observations**:
- $N \coloneqq |\mathcal{E}| = O(\log(\eta_{\max}/\eta_{\min})) = O(\log T)$.
- $\eta_k \leq \eta^* \leq 2\eta_k$ for some $k \in \{1, 2, \ldots, N\}$.

Let $w_t^1, \ldots, w_t^N \in \mathcal{W}$ be the outputs of OGD with rates $\eta_1, \ldots, \eta_N$.

For every expert $i$, we have

$$\sum_{t=1}^T f_t(w_t^i) - \sum_{t=1}^T f_t(u_t) \lesssim \frac{1 + P_T}{\eta_i} + \eta_i G^2 T.$$

### Expert Problem Formulation
- There are $N$ experts $i = 1, 2, \ldots, N$.
- At round $t$, expert $i$ incurs loss $\ell_{t,i} \in [0, H]$.

We compute $p_t \in \triangle^N$ based on history $\ell_1, \ldots, \ell_{t-1} \in [0, H]^N$ to compete against any expert $i^*$ in expectation:

$$\text{Regret}(i^*) = \sum_{t=1}^T \langle p_t, \ell_t \rangle - \sum_{t=1}^T \ell_{t,i^*}.$$

### Hedge Algorithm
Assign smaller probabilities to experts with larger cumulative losses:

$$w_{t,i} \propto \exp(-\epsilon \sum_{s=1}^{t-1} \ell_{s,i}),$$ 

where $\epsilon = \frac{1}{H}\sqrt{\frac{\log N}{T}}.$

The regret against any expert $i^*$ is bounded as follows:

$$\text{Regret}(i^*) = \sum_{t=1}^T \langle p_t, \ell_t \rangle - \sum_{t=1}^T \ell_{t,i^*} \lesssim \sqrt{T \log N}$$

See, e.g., [Hazan's book](https://arxiv.org/abs/1909.05207) for the proof (if there is a perfect expert, the $\log N$ regret is possible via bisection).

### Combining for Dynamic Regret Bound
At each $t$:
- Expert $i$ computes $w_t^i$ by OGD with $\eta_i$.
- Learner outputs $w_t = \sum_{i=1}^N p_{t,i} w_t^i$.

Define $\ell_{t,i} \coloneqq \langle w_t^i, g_t \rangle + G \in [0, 2G]$, where $g_t = \nabla f_t(w_t)$.

For every $i^*$, by convexity, 

$$\sum_{t=1}^T f_t(w_t) - \sum_{t=1}^T f_t(w_t^{i^*}) \leq \sum_{t=1}^T \langle g_t, w_t - w_t^{i^*} \rangle.$$

If $p_t \in \Delta^{N}$ is computed by Hedge, the r.h.s. is bounded by

$$\sum_{t=1}^T \langle g_t, \sum_i p_{t,i} w_t^i  \rangle - \sum_{t=1}^T \langle g_t, w_t^{i^*} \rangle = \sum_{t=1}^T \langle p_t, \ell_t \rangle - \sum_{t=1}^T \ell_{t,i^*} \lesssim G\sqrt{T \log N}.$$

In particular, this is true for $i^* = k \in \mathcal{E}$ such that $\eta_k \leq \eta^* \leq 2\eta_k$. 

- **Meta-regret against expert $k$**: Recalling $N = O(\log T)$, 

$$\sum_{t=1}^T f_t(w_t) - \sum_{t=1}^T f_t(w_t^k) \lesssim G\sqrt{T \log \log T}.$$

- **Expert $k$'s regret**: Recalling $\eta^* = \frac{1}{G}\sqrt{\frac{1 + P_T}{T}}$, 

$$\sum_{t=1}^T f_t(w_t^k) - \sum_{t=1}^T f_t(u_t) \lesssim \frac{1 + P_T}{\eta_k} + \eta_k G^2 T \le \frac{1 + P_T}{\eta^*/2} + \eta^* G^2 T \lesssim G\sqrt{T(1 + P_T)}.$$

Therefore, ignoring $\log\log T$, we obtain

$$\sum_{t=1}^T f_t(w_t) - \sum_{t=1}^T f_t(u_t) \lesssim G\sqrt{T(1 + P_T)}.$$

## References and Related Topics

**Reference**:

[Zhang et al., NeurIPS '18: "Adaptive online learning in dynamic environments"](https://papers.nips.cc/paper_files/paper/2018/hash/10a5ab2db37feedfdeaab192ead4ac0e-Abstract.html)

Their proof eliminates the $\log\log T$ factor by a more careful analysis of the Hedge algorithm with a more sophisticated choice of initial weights.

**Other topics on learning rate tuning**:
- AdaGrad
- Universal OCO
- Parameter-free OCO
- The Road Less Scheduled
- Gradient Descent: The Ultimate Optimizer