# WorldGen-Voxel-DMGibbs
Another attempt for voxel world generation.

(Simply copied from my notebook, do expect weird stuffs)  
# Proposed method

>Same idea: https://arxiv.org/pdf/1901.08508.pdf Section 3 :rofl:

Consider the KL divergence between a measure $q$ and target unknown measure $p$:
$$
\begin{aligned}
D(q \parallel p)
&= - \mathbb{E}_q \left[ \log \frac{p(x)}{q(x)} \right] & \\
&= - \mathbb{E}_q \left[ \log p(x) \right] - \mathbb{H}[q(x)] & \\
&= \mathbb{E}_q \left[ Z + \beta E(x) \right] - \mathbb{H}[q(x)] &(1)
\end{aligned}
$$
where $\mathbb{H}$ denotes the information entropy, and recall that $p$ is given by $p(x) \propto e^{- \beta E(x)}$ with normalizing constant $Z = \int_x e^{- \beta E(x)}$.
By minimizing eq.1 we can approximate $p$ by $q$.

We'd like to approximate $p$ using $q(x, \beta)$.
Being inspired by the recent success of diffusion models, during the generation process, we want to gradually increase $\beta$. Denote the schedule function as:
$$
\beta(t) : \mathbb{R} \rightarrow \mathbb{R}
$$
which maps from time $t$ to the temperature constant $\beta$.

## Entropy estimation
One major challange in eq.1 is to estimate the entropy $\mathbb{H}[q(x)]$. Specifially, we'd like to borrow the idea in MINE (Belghazi+ 18'). First, given an reference distribution $u(x)$, observe that:
$$
D(q || u) = \mathbb{E}_q[u(x)] - \mathbb{H}[q(x)]
$$
And by furthermore letting $u$ be uniform distribution, we have:
$$
D(q || u) = const - \mathbb{H}[q(x)]
$$
In MINE, the *Donsker-Varadhan representation of KL divergence* $D$ is used to obtain a lower bound on KL:
$$
D(q || u) \geq \sup_{h \in \mathcal{F}} \mathbb{E}_q\left[h(x)\right] - \log \left(\mathbb{E}_u\left[e^{h(x)}\right]\right)
$$
>note that they have also used an exponential moving average to stablize the gradient estimation. We will also apply that trick in our method.

We can use this lower bound to establish an upper bound on entropy $\mathbb{H}[q(x)]$:
$$
\begin{aligned}
\mathbb{H}[q(x)]
&= const - D(q \parallel u) \\
&\leq const - \mathbb{E}_q\left[h(x)\right] + \log \left(\mathbb{E}_u\left[e^{h(x)}\right]\right)
\end{aligned}
$$

> An lower bound on entropy is more preferrable (to get rid of adversarial training), yet I couldn't found suitable results to estimate it. This will be a major direction in the future work. Besides, instead of an accurate estimation of $\mathbb{H}[q(x)]$, we'd rather more interested in an accurate estimation of $\nabla \mathbb{H}[q(x)]$ (its gradient).

Thus, we'd like to employ an adversarial training scheme for maximizing the estimated entropy in our objective.

## Training the network
### Notations & parameters
- $T$ - Fixed timestep count
- $t = 0, 1, 2, \dots, T$ - Discrete timesteps
	- $dt = 1$
- $\mathcal{X}$ - Data space (discrete, space of token sequences)
	- $x_t$ - Data at time $t$
- $\beta(t)$ - Temperature scheduling function
- $S_t$ - Data buffer of time $t$
### Formalization
our network is composed by those components:
- Network $f(x, \beta(t); \theta): (\mathcal{X}, \mathbb{R}) \rightarrow \mathcal{X}$, which estimates the forward advancement given input $x$ at time $t$.
	- Denote the distribution of its outputs as $q_\beta (x)$.
- Network $g(x; \phi) : (\mathcal{X}, \mathbb{R}) \rightarrow \mathbb{R}$, estimates the energy (score) $E(x)$.
- Network $h(x, \beta(t);\phi) : (\mathcal{X}, \mathbb{R}) \rightarrow \mathbb{R}$, estimates the entropy $\mathbb{H}[q_\beta(x)]$.

Objective for $f$:
$$
L_{f} = h(f(x)) + \mathbb{E}\left[\beta\left(t + dt\right) \cdot g\left(f(x)\right)\right]
$$
Objective for $g$ (MSE / regression):
$$
L_g = \frac{1}{2} \left( g(x) - E(\tilde{x}) \right)^2
$$
Objective for $h$:
$$
L_h = \mathbb{E}[h(x_t)] - \log \left( \mathbb{E}[e^{h(x_0)}] \right)
$$
> For now, we use a single $h$ for all $t$ and no other corrections. This is obiviously wrong but we will just keep it like this for now and left it to future work ...

**The training is proceeded as follows:**
1. Generate random data from uniform distribution $u$ as $x_0$ and add it to the data buffer $S_0$.
2. Pick a timestep $t$ randomly, then pick $x \in S_t$. (forms a minibatch; omitted for clearity)
3. Obtain $x' = f(x, \beta(t); \theta)$, minimize $L_f$.
4. minimize $L_g$, maximize $L_h$, and update the EMA in $h$ (see MINE Sec 3.2).

During inference, we simply generate a new $x_0$ from $u$, then feed it into the network $f$, iteratively for $T$ times.
