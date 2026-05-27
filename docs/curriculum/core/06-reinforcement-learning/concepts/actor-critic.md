---
title: Actor-Critic
slug: actor-critic
layer: core
subject: 06-reinforcement-learning
page_type: concept
state: drafted
authors_anchored: [sutton, barto, mnih, silver]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [policy-gradient, value-functions, markov-decision-processes]
tags: [reinforcement-learning, policy-gradient, variance-reduction, rl-optimization, actor-critic]
updated: 2024-11-05
has_mvb: true
---

# Actor-Critic

Imagine the stand-up comedian who can no longer wait for the final applause. If she treated the audience feedback like a single deferred reward, she would perform the whole set, gather the applause at the end, and then try to guess which jokes to keep; each update would be noisy, delayed, and hard to interpret—just like the REINFORCE episode returns that blend dozens of stochastic choices. Instead, a savvy comedian listens to the chuckles and groans in real time, letting those micro-reactions shape the next sentence while the set is still in motion. That’s the basic difference that actor-critic architectures exploit: the actor makes the creative choices while a critic evaluates the current slice of the performance, shrinking the variance of feedback and turning chaotic trial-and-error into a directed optimization. By the end of this page, you will be able to explain how that critic estimate stays compatible with the policy gradient, how temporal-difference learning keeps both networks stable, how asynchronous or off-policy variants scale to millions of steps, and how to implement a working Advantage Actor-Critic (A2C) agent that can solve CartPole-v1 in PyTorch on a free Colab CPU.

## The territory

Reinforcement learning (RL) lives between two extremes. On one end, value-based algorithms like Q-learning ignore the stochasticity of the actor by learning action-values and picking whichever action currently appears best. On the other end, pure policy gradients, epitomized by REINFORCE, rely on sampled trajectories and weight each visited action by the total return from that trajectory, which yields unbiased gradients but high variance that explodes with horizon length. Actor-critic sits squarely between them. It keeps the policy gradient objective intact—the actor still updates via expectation over \(\nabla_\theta \log \pi_\theta(a \mid s)\)—but replaces the noisy episode-level return with a state-dependent baseline provided by the critic. That critic can be a value function \(V^\pi(s)\) estimating the expected return from state \(s\) or an action-value \(Q^\pi(s,a)\), and it learns from temporal-difference (TD) errors so that the actor’s update uses more localized, lower-variance feedback. Early applications of actor-critic, such as the elevator control system in Crites & Barto (1995) [http://all.cs.umass.edu/pubs/1995_96/crites_b_95.pdf], already demonstrated how decoupling policy and value learning stabilizes control in large, nonstationary systems. The big picture is that actor-critic inherits the expressive policies of policy gradients and the variance-reducing baselines of value methods, which is why every modern RL algorithm from A2C to PPO to soft actor-critic is built around that dual network structure. How does that happen, mathematically and in code, and what keeps the two networks from undoing each other’s work?

## How it works

The policy gradient theorem provides the starting point. When optimizing a parametrized policy \(\pi_\theta(a \mid s)\), the policy gradient objective is

\[
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta(\cdot \mid s)}\big[\nabla_\theta \log \pi_\theta(a \mid s) \, Q^{\pi}(s, a)\big],
\]

where \(d^\pi(s)\) is the stationary distribution over states under the current policy, \(Q^{\pi}(s, a)\) is the expected return after taking action \(a\) in state \(s\), and the expectation ranges over on-policy transitions. The actor-critic insight is to replace \(Q^{\pi}(s,a)\) with a learned estimate and to subtract a baseline without biasing the gradient. When the critic estimates the state-value \(V_\phi(s)\), the advantage \(A^{\pi}(s,a) = Q^{\pi}(s,a) - V^\pi(s)\) quantifies how much better the chosen action was than the average. During learning, the actor update becomes

\[
\nabla_\theta J(\theta) \approx \mathbb{E}\big[\nabla_\theta \log \pi_\theta(a \mid s) \, (\delta_t)\big],
\]

where \(\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)\) is the TD error. Here \(r_t\) is the reward at timestep \(t\), \(\gamma\) is the discount factor, \(s_{t+1}\) is the next state, and \(V_\phi\) is the critic network. The TD error measures how surprised the critic is; positive surprise tells the actor the selected action led to better-than-expected outcomes. Because \(\mathbb{E}[\nabla_\theta \log \pi_\theta(a \mid s) V_\phi(s)] = 0\), subtracting the critic adds no bias while cutting the variance that would come from using the full Monte Carlo return. The actor still points in the right direction, but the direction now leans on a local evaluation rather than a long delayed sum.

### Critic learning and compatibility

The critic itself learns by minimizing squared TD error:

\[
\mathcal{L}_\text{critic}(\phi) = \mathbb{E}\big[(r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t))^2\big],
\]

where the expectation covers transitions drawn either on-policy or from a behavior policy. Every sample provides a bootstrapped target \(r_t + \gamma V_\phi(s_{t+1})\). Early work by Konda & Tsitsiklis (2000) [https://www.jmlr.org/papers/volume1/konda00a/konda00a.pdf] showed that, for convergence, the critic must update on a faster timescale than the actor so that the actor sees a nearly stationary value function as it adjusts. They framed the critic as solving a projected Bellman equation and proved convergence when the actor’s updates are small enough, leading to the multi-timescale scheme widely used today: \(\phi\) updates with a stepsize \(\beta_t\) and \(\theta\) updates with a smaller stepsize \(\alpha_t\), and \(\alpha_t/\beta_t \to 0\). This separation lets the critic track the value landscape while the actor explores slowly around it.

In practice, compatibility is also important. When the critic shares a parameterization with the actor or is trained on the same feature space, the TD error becomes a compatible advantage estimator, as described by Konda & Tsitsiklis, meaning the inner product \(\nabla_\theta \log \pi_\theta(a \mid s) \cdot \delta_t\) approximates the true advantage. When compatibility fails—say, when the critic and actor use entirely separate architectures—the actor can receive misleading signals, and training drifts. This is why modern implementations often tie the bottom layers of the actor and critic or use shared encoders before branching into the separate heads.

### Variance reduction through temporal difference

The TD error is where the critic provides variance reduction. Without a critic, Monte Carlo returns in REINFORCE can have variance proportional to episode length because each action's gradient depends on the entire future reward stream. With a critic, the actor sees only the bootstrapped difference \(r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)\), which depends on a single reward and two value estimates. The variance of that difference is bounded even for long horizons because each update reuses overlapping transitions. The critic does introduce bias through bootstrapping, but that bias is often smaller than the variance saved, and it can be controlled by tuning the TD learning rate \(\beta_t\). The bias-variance trade-off is the fundamental equilibrium that actor-critic architectures solve: the critic provides a biased but low-variance signal, and the actor averages those signals to maintain unbiased gradients thanks to the policy gradient theorem.

### Stabilizing the duo: trust regions, entropy, and target networks

The actor and critic must also avoid destabilizing each other. When the critic overfits or its targets shift explosively, the actor chases phantom gradients. PPO (Schulman et al. 2017) [https://arxiv.org/pdf/1711.04755] introduced clipping on the policy ratio \(\frac{\pi_\theta(a \mid s)}{\pi_{\theta_\text{old}}(a \mid s)}\) so that individual actor steps stay within a trust region, preventing sudden policy jumps from invalidating the critic’s assumption. PPO also adds an entropy bonus

\[
\mathcal{L}_\text{entropy} = -\beta \mathbb{E}[\mathcal{H}(\pi_\theta(\cdot \mid s))],
\]

where \(\mathcal{H}\) is the Shannon entropy and \(\beta\) controls how much exploration the actor keeps. This penalty ensures the actor keeps sampling diverse actions long enough for the critic to gather informative TD errors.

Target networks or delayed value updates, borrowed from deep Q-learning, also appear in actor-critic stacks. Instead of using the latest critic estimate in the TD target, some implementations maintain a slowly moving average \(V_{\phi^-}\) and compute

\[
\delta_t = r_t + \gamma V_{\phi^-}(s_{t+1}) - V_\phi(s_t),
\]

where \(\phi^-\) tracks \(\phi\) via Polyak averaging. This decouples the target from the critic’s instantaneous parameters and smooths the bootstrapping signal, which is why many off-policy actor-critic algorithms use such targets.

### Asynchrony and parallelism

Scaling actor-critic to deep neural networks required overcoming the data correlation that appears when a single agent runs on-policy for long. Mnih et al. (2016) [https://arxiv.org/abs/1602.01783] introduced Asynchronous Advantage Actor-Critic (A3C), where multiple worker threads simultaneously interact with copies of the environment, accumulate gradients over a few steps, and asynchronously update shared actor and critic weights on a central parameter server. Each worker computes its own TD advantage

\[
\hat{A}_t = \sum_{i=0}^{k-1} \gamma^i r_{t+i} + \gamma^k V_\phi(s_{t+k}) - V_\phi(s_t),
\]

where \(k\) is the rollout length before the worker posts gradients. Because the workers explore different parts of the state space in parallel, the shared critic sees decorrelated data without needing experience replay. Mnih et al. also found that reducing gradient variance by accumulating \(k\)-step returns (\(k\) typically 5) stabilizes learning even with the asynchronous noise. The result is an on-policy actor-critic algorithm that can train Atari agents in a fraction of the time earlier synchronous methods needed.

### Off-policy critics and behavior policies

The straight actor-critic setup is on-policy, but many real systems benefit from reusing off-policy data. Degris, White, and Sutton (2012) [https://arxiv.org/abs/1205.4839] derived off-policy actor-critic updates by introducing importance sampling ratios \(\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\mu(a_t \mid s_t)}\), where \(\mu\) is the behavior policy that generated the data. The actor gradient becomes

\[
\nabla_\theta J(\theta) \approx \mathbb{E}\big[\rho_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) Q^\pi(s_t, a_t)\big],
\]

and the critic is trained via TD on the same transitions, but now the expectation is taken over \(\mu\). Importance sampling can hurt variance, so Degris et al. introduce eligibility traces and truncated ratios to keep the updates stable. The advantage is that replay buffers populated by old policy data can be leveraged, which is critical in real-world robotics or offline RL where collecting fresh on-policy data is expensive.

### Modern actor-critic in RLHF

Actor-critic also underpins modern RLHF pipelines, but the critic often needs to operate on large language models and long decision chains, where storing per-token rewards or samples is infeasible. Shao et al. (2024) [https://arxiv.org/abs/2405.08422] introduced Group Relative Policy Optimization (GRPO), which replaces the single critic network with a lightweight reward head that computes group-relative averages over pre-collected reward model outputs. By aggregating preferences across user-specified groups, GRPO reduces memory pressure while still supplying the actor with a low-variance signal resembling a critic advantage. The policy update resembles standard actor-critic but uses the group-average reward \(\bar{R}_g\) as a baseline:

\[
\delta_t = R_t - \bar{R}_{g(t)} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t),
\]

where group \(g(t)\) indexes which preference cluster the current response belongs to. GRPO shows that even when the critic cannot evaluate every token directly, actor-critic remains meaningful as long as some summary statistic captures the relative quality, which is the principle that guides RLHF deployments at scale.

### Summary of mechanisms

In summary, actor-critic replaces full return estimation with a learned critic, uses TD error as an advantage, stabilizes training via trust regions and entropy, scales via asynchronous workers or off-policy replay, and extends to modern RLHF through group-relative reward baselines. Each variant—the synchronous Advantage Actor-Critic (A2C), asynchronous A3C, PPO, off-policy actor-critic, or GRPO—follows the same skeleton: an actor proposing actions, a critic evaluating them, and a carefully synchronized update rule so that the variance reduction of the critic dominates the small bias introduced by bootstrapping.

## Where the field is now

On the research frontier, GRPO (Shao et al. 2024) sharpened the actor-critic lens for RLHF by showing that the critic can be replaced with aggregated group-relative rewards without losing the essential variance reduction. This freed practitioners from maintaining per-token global reward buffers and enabled faster partial updates during preference learning. The engineering frontier keeps actor-critic central too: OpenAI’s RLHF blog (OpenAI 2024) explains that PPO, an actor-critic method with clipped ratios, is the workhorse for aligning large language models because it balances high-throughput training with stability, and the entire pipeline—from reward modeling to policy optimization—depends on the critic to deliver gradients that respect human preferences without exploding. PPO’s design, which adds a loss term \(\mathcal{L}_\text{clip} = \mathbb{E}[\min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t)]\) where \(r_t\) is the probability ratio, is why RLHF in production can safely proceed with large batches of autoregressive tokens without destabilizing the policy.

A small table summarizing the contemporary landscape helps make this concrete.

| Algorithm | Variance control | Parallelism | Production touchpoint |
|-----------|------------------|-------------|-----------------------|
| A2C | TD advantage baseline | Single-actor | Training small robotics prototypes |
| A3C | Multiple workers reduce correlation | Async workers | Early Atari and robotics labs |
| PPO | Clipped ratio + entropy bonus | Batch mini-updates | OpenAI RLHF alignment pipeline |
| GRPO | Group-relative reward baseline | Data aggregation | RLHF with token-level reward constraints |

The research frontier question is whether actor-critic can keep reducing variance while handling extremely long horizons, like those encountered in reasoning chains or multi-agent simulation. The production frontier is set by how quickly an actor-critic policy can be retrained as the critic (or reward model) changes, because real systems like RLHF deployments retrain multiple times per week, and each retraining must end without destabilizing the behavior policy. That’s why system teams monitor the critic’s loss, the activation norms, and the PPO ratio histograms to ensure the dual networks remain synchronized.

## What's still open

Can actor-critic be made to perform mathematically rigorous, token-level credit assignment in reasoning tasks where the reward is sparse, the chain is long, and labeling each reasoning step is impractical? In such settings, the critic must hallucinate intermediate value estimates from partial context, and any misestimation can send the actor toward reward hacking. A second open question is whether group-relative baselines like GRPO can be generalized beyond preference clusters to continuous latent spaces, enabling the critic to operate on embeddings instead of discrete reward groups. Third, does asynchronous actor-critic still dominate once we add noisy oracles into the mix—are there convergent update rules when each worker’s critic is allowed to drift before its gradients are aggregated? Each of these questions can seed a research sprint: formalizing token-level value functions, proving convergence with latent group summaries, and designing new synchronization strategies for noisy critics.

## Where to read next

If you want the probabilistic foundation that makes policy gradients sound, → [Policy gradient](policy-gradient.md) explains why the log-derivative trick yields unbiased gradients and how baselines cancel variance. The engineering counterpart is → [[value-functions]], which shows how TD learning and bootstrapping work for the critic you just implemented. For the RLHF story beyond PPO and GRPO, → [[reward-modeling]] covers how human feedback is collected and converted into the reward signals consumed by actor-critic optimizers.

## Build it

Training Advantage Actor-Critic from scratch on CartPole verifies you understand how the actor, critic, TD error, and rollout intertwine.

**What you're building:** A PyTorch Advantage Actor-Critic agent that solves `CartPole-v1` in Gym, running on a free Colab CPU in under two minutes so you can see the dual networks and TD advantage in action.

**Why this is valuable:** The build forces you to instantiate both networks, compute the TD error \(\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\), and update the actor with the resulting advantage instead of the episode return, making the variance-reduction mechanics explicit.

**Stack:**
- **Model:** Custom actor-critic network—two separate heads sharing a three-layer MLP (no external HF model required).
- **Dataset:** `CartPole-v1` from Gym (provided by `gymnasium==0.28.1`, environment length ≤ 500).
- **Framework:** PyTorch 2.1 (with `torch.optim.AdamW`), Gymnasium, NumPy.
- **Compute:** Free Colab CPU (2 vCPUs, ≤2 minutes per run).

**The recipe:**
1. Install the stack with `pip install torch==2.1.0 gymnasium==0.28.1 torchtyping`, and seed both NumPy and PyTorch for reproducible runs.
2. Create the CartPole environment, normalize observations if desired, and batch transitions into mini-rollouts of 5 steps to compute multi-step returns.
3. Define the actor-critic network: shared MLP to 128 units, then separate linear heads for policy logits and value \(V_\phi(s)\). Use AdamW with actor learning rate \(1e-4\) and critic learning rate \(5e-4\); inside the training loop, compute \(\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)\) and the policy loss \(-\log \pi_\theta(a_t \mid s_t) \cdot \delta_t\).
4. Evaluate by running 20 test episodes every 1000 steps and report the average return; expect returns exceeding 475 within 5k gradient updates.
5. What you now have is a checkpointed actor-critic policy that solves CartPole with a stable TD advantage, plus logs of \(V_\phi\) vs. Monte Carlo returns showing the critic’s accuracy.

**Expected outcome:** A working A2C checkpoint and plotted learning curves proving that the TD error drives the actor toward stable behavior.

- **CS student:** Run the same recipe on Colab but log the critic loss and plot the advantage estimates per step to see how the critic de-noises the reward.
- **Applied engineer:** Quantize the actor’s policy head to int8 with PyTorch’s static quantization, wrap the policy in a simple Flask app, and target a 50 ms inference latency on an A10 instance.
- **Applied researcher:** Replace the TD error with a truncated importance-sampled variant from Degris et al. (2012) and compare convergence curves to test whether off-policy data accelerates learning.
- **Frontier researcher:** Extend GRPO’s group-relative baseline to token-level reasoning by keeping per-token reward buckets and defining a falsification test: if the actor’s policy gradient improves while the critic’s bucketed variance increases, the baseline is insufficient.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*