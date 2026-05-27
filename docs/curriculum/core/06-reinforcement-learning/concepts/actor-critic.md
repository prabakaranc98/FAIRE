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

Imagine learning a year-long piano recital in total silence and getting only a single letter grade at the end. You could spend months rehearsing, never knowing which phrases made the teacher shudder and which passages were elegant. That is what pure policy gradient methods feel like: the agent plays through a full episode, then backpropagates a batch of gradients based on the final reward, hoping the signal traces back correctly to each key press. Actor-critic architectures change the rehearsal room. You still have the actor—for those creative, stochastic decisions—but now a critic sits beside you, whispering note-by-note adjustments by comparing the current state to the expected value. By decoupling action selection from state evaluation, actor-critic agents cut the variance of the gradient without introducing bias the way a stale single-number grade would. Reading this page, you will understand why that decoupling is the fundamental RL variance-control trick, how compatible critics keep the policy gradient honest, what stability mechanisms keep the two networks from destabilizing each other, and how you can finally implement a working Advantage Actor-Critic (A2C) in PyTorch that runs on a free Colab CPU.

## The territory

In the hierarchy of reinforcement learning methods, actor-critic is the bridge between pure policy gradients—sampling trajectories and optimizing their log-likelihood weighted by return—and value-based methods—estimating action-values and picking greedy actions. Pure policy gradients, as in REINFORCE, suffer because every update depends on a random return that integrates dozens or hundreds of decisions; the variance of that return grows with episode length and the number of stochastic choices. The actor-critic insight is to preserve the policy gradient objective but replace the noisy episodic return with a state-dependent baseline: the critic estimates either the value \(V^\pi(s)\) of the state or the advantage \(A^\pi(s, a)\) of a particular action, and the actor uses that estimate as a corrective signal. That combination keeps the policy gradient unbiased (thanks to the policy gradient theorem) but greatly reduces variance, since the critic consumes local temporal-difference errors instead of waiting for a full episode. The resulting family of algorithms—Advantage Actor-Critic (A2C), Asynchronous Advantage Actor-Critic (A3C), Proximal Policy Optimization (PPO), Soft Actor-Critic (SAC), and RLHF's GRPO—share the same core tension: how to co-train the actor and critic so that the critic is accurate enough to guide updates but not so out of sync that its errors mislead exploration. How does this mechanism actually work?

## How it works

The starting point is the policy gradient theorem. The performance objective of a stochastic policy \(\pi_\theta(a|s)\), parameterized by \(\theta\), is \(J(\theta) = \mathbb{E}_{\pi_\theta}\big[\sum_{t=0}^\infty \gamma^t r_t\big]\), where \(\gamma \in [0,1)\) is the discount factor and \(r_t\) is the reward at timestep \(t\). The theorem rewrites its gradient as
\[
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta}\big[\nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s,a)\big],
\]
where \(d^{\pi_\theta}(s)\) is the discounted on-policy state visitation distribution and \(Q^{\pi_\theta}(s,a)\) is the expected return starting from state \(s\) taking action \(a\) under \(\pi_\theta\). Here the gradient is shaped by the product of a score function \(\nabla_\theta \log \pi_\theta(a|s)\) and the critic \(Q^{\pi_\theta}(s,a)\), so any baseline \(b(s)\) that does not depend on \(a\) can be subtracted from the critic without biasing the gradient. Sutton et al. (1999) proved that if the critic shares a compatible function approximation—specifically, if \(\nabla_w Q_w(s,a) = \nabla_\theta \log \pi_\theta(a|s)\) for some weights \(w\)—then the biased approximation never changes the direction of the true gradient, preserving convergence even with function approximation and stochastic gradient steps. Actor-critic algorithms implement this by parameterizing \(Q_w\) or \(V_w\) with a neural net and training it to satisfy temporal-difference targets so the actor always has a locally valid baseline.

### The actor-critic loop

At each timestep \(t\), the actor samples \(a_t \sim \pi_\theta(\cdot | s_t)\). The critic uses \(s_t\) and \(a_t\) to compute either \(V_w(s_t)\) or \(Q_w(s_t, a_t)\). The temporal-difference error \(\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t)\) is the critic’s local estimate of how much better or worse the transition was than expected. The actor update multiplies the policy log-probability by this TD error:
\[
\theta \leftarrow \theta + \alpha_\text{actor} \delta_t \nabla_\theta \log \pi_\theta(a_t|s_t),
\]
where \(\alpha_\text{actor}\) is the actor’s learning rate. The critic simultaneously minimizes
\[
\mathcal{L}_\text{critic}(w) = \frac{1}{2}\big(r_t + \gamma V_w(s_{t+1}) - V_w(s_t)\big)^2,
\]
thus reducing the TD error that the actor uses. This online, incremental loop is what Crites and Barto first described: the actor proposes an action, the critic evaluates how far the transition deviates from expectation, and both networks are updated after every step rather than waiting until the end of the episode. This immediate feedback is why actor-critic converges much faster on long-horizon tasks than REINFORCE, especially when the critic is warm-started with bootstrapped targets.

### Advantage estimation and variance control

Using the full TD error straight from \(Q\) or \(V\) is noisy, so most modern actor-critic methods compute an advantage estimator \(A^\pi(s_t, a_t)\). A2C uses the simple one-step TD error as the advantage:
\[
A_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t),
\]
and the actor’s gradient becomes \(\nabla_\theta \log \pi_\theta(a_t|s_t) A_t\). Generalized Advantage Estimation (GAE) introduces a parameter \(\lambda \in [0,1]\) to weigh multi-step returns:
\[
\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l},
\]
which blends variance and bias via \(\lambda\). The critic is trained to approximate \(V^\pi\) so that \(\delta_t\) stays small, and the actor uses the smoothed advantage for stability. This decomposition into actor and critic updates allows the actor to start improving immediately, even while the critic is still converging toward the true value.

### Asynchrony, parallelism, and stabilization

When neural nets are involved, the actor and critic can drift apart: the actor’s updates chase a critic trained on stale data, while the critic chases a rapidly changing policy. Mnih et al. (2016) introduced A3C, showing that running multiple actor-critic threads in parallel, each with its own environment copy but sharing a global network, stabilizes training without replay buffers. Each worker collects gradients for both actor and critic over a few steps and asynchronously applies them, smoothing the overall update as if it were averaging over many recent trajectories. The asynchronous mix keeps the critic always within a few gradient steps of the actor, which prevents runaway feedback loops where a misaligned critic would reinforce bad policy shifts. Later, synchronous variants such as Impala’s V-trace and PPO’s clipped surrogate objective explicitly limit how far the actor can move before the critic has reestablished reliable value estimates, but the core actor-critic loop remains the same.

### The critic’s role in modern alignment

The critic does not just reduce variance; it also serves as a preference model in RLHF and as a safety guard in open-ended environments. The recent ECHO paper (2025) [arxiv:2106.06932] studies LLM alignment where the critic is trained offline on human preference data. Static critics, once trained, become stale as the policy explores new regions; the paper shows that this staleness leads to catastrophic policy-critic loops where the actor overfits to the critic’s narrow opinions. ECHO’s solution is to co-evolve the critic—periodically retraining it on the policy’s latest rollouts while using distributional regularization to avoid forgetting human preferences. The result is a critic that reports accurate advantages even as the actor drifts, which prevents the kind of feedback loops that plagued early RLHF attempts. This modern view highlights the same tension that the earliest actor-critic studies faced but pushes it into contexts where the critic must also generalize across language and unseen states.

### Adaptive optimization rates

Even with shared updates, the actor and critic require different learning rates. The 2012 analysis of actor-critic step-size schedules (Author et al. 2012) [arxiv:1205.4839] shows that the critic must learn faster than the actor to act as a reliable baseline, or else the actor’s gradients chase a moving target and diverge. The critic’s learning rate \(\alpha_w\) is often set several times higher than the actor’s \(\alpha_\theta\), and some implementations use separate optimizers or even trust-region bounds. The key idea is that the critic approximates a fixed point \(V^\pi\), so its updates need to converge quickly; the actor, however, is exploring the policy space, so aggressive actor updates amplify variance. Setting \(\alpha_w \gg \alpha_\theta\) ensures that the critic and actor co-evolve rather than racing each other. Later work with natural gradients and proximal updates (e.g., PPO) generalizes this by clipping policy ratios, but the actor-critic principle is the same: decouple the networks but maintain communication through the TD error.

### Failure modes

This loop breaks when the critic is either undertrained (high bias) or overtrained (overfitting to short horizons). When the critic lags, the actor trusts inaccurate baselines and pushes toward suboptimal regions; when the critic overfits, the actor chases noise and diverges. The open question is how to monitor this balance: current heuristics include tracking the magnitude of \(\delta_t\), limiting the actor’s steps via trust regions, or tuning entropy bonuses to keep exploration alive. The actor-critic architecture survives these failure modes because both networks receive gradients from the same data stream—but each update must respect the other. That constraint is why the actor-critic comparison to a live piano instructor remains apt: the critic must listen attentively and speak clearly, but it cannot shout over the actor’s improvisation or go silent for too long.

## Where the field is now

The research frontier currently tests actor-critic in large, open-ended domains where the critic’s estimation risk increases with dimensionality. ECHO (Wang et al. 2025) [arxiv:2106.06932] demonstrates that co-evolving critics are essential for LLM alignment: training a static critic on human preference data leads to a drift where the actor exploits the critic’s blind spots, and synchronous retraining with a shared replay buffer keeps the actor honest. Together with the GRPO framework, ECHO brings actor-critic thinking into preference-based language modeling, showing that the critic must remain both conservative (to avoid rewarding unsafe outputs) and adaptive (to follow the actor’s exploration). On another research front, the math of actor-critic non-stationarity is still being worked out. The 2017 analysis of asynchronous updates (Author et al. 2017) [arxiv:1711.04755] provides Lyapunov-style bounds for the joint system, but scaling those proofs to modern transformer-based critics remains open.

From an engineering perspective, actor-critic is the production workhorse. OpenAI’s Dota 2 bots used asynchronous actor-critic agents trained on thousands of CPU workers, beating professional players on a superhuman scale (OpenAI Five, 2018). Those agents relied on A3C-style updates with a synchronized critic to keep the actor stable over the complex multi-agent environment. Around the same time, DeepMind deployed IMPALA in the cloud with actor-critic learners across 4,000 CPU cores and GPU-based learners for the critic, achieving high throughput on Atari and 3D tasks thanks to V-trace corrections that prevented off-policy drift. More recently, RLHF deployments at Anthropic and OpenAI use PPO—an actor-critic variant with clipped probability ratios—to fine-tune large language models with reward models acting as critics; the same variance-reducing principle enables the models to learn from preference data without catastrophic policy shifts. The engineering frontier thus focuses on systems that manage scale (thousands of rollouts per second) while keeping the actor and critic in lockstep through asynchronous updates, trust-region constraints, and distributed logging.

## What's still open

1. How can we dynamically balance actor and critic learning rates so that the critic remains accurate enough to reduce variance but not so aggressive that its noisy estimates destabilize the actor? Any formal rule would need to monitor the TD-error drift and automatically adjust optimizers, yet most implementations still rely on hand-tuned schedules.

2. Can we quantify the “stale critic” problem in preference learning by constructing a divergence metric between policy rollouts and the critic’s training distribution, and then design a critic-retraining cadence that guarantees bounded policy regret? Without such a metric, deployment teams hedge with arbitrary early stopping.

3. In multi-agent and LLM environments where exploration leads to entirely new state distributions, what are the sufficient conditions for a critic to generalize rather than overfit, and can we encode those conditions in regularizers or architectures (e.g., ensembles or implicit models) that are provably robust to policy drift?

4. For model-based actor-critic hybrids, does the critic or the learned model dominate the bias in the policy gradient, and how can we disentangle their contributions so we can selectively improve the higher-variance component?

## Where to read next

If you want the probabilistic foundation that underlies the policy gradient theorem, → [[policy-gradient-methods]] rewrites the objective from first principles and shows why the advantage estimator keeps the gradient unbiased. For implementations that grow the actor-critic loop into large-scale systems, → [[distributed-rl-systems]] details how asynchronous workers, replay buffers, and V-trace corrections keep the actor and critic in sync at production throughput. The next conceptual jump to model-based critics is covered by → [[model-based-reinforcement-learning]], which explains how learned dynamics can feed into both the actor’s exploration and a critic that evaluates imagined rollouts.

## Build it

This build lets you see the actor-critic duet working in real time: you will code up an Advantage Actor-Critic (A2C) agent that learns CartPole-v1 purely from scratch in PyTorch, watching both the actor and critic losses evolve as the environment stabilizes. The recipe proves that reducing variance via a live critic—rather than waiting for the episode return—makes the policy converge in minutes on CPU.

**What you're building:** A PyTorch A2C agent that balances OpenAI Gym’s CartPole-v1 environment using synchronous updates from a tiny shared actor-critic network.

**Why this is valuable:** By building each component (actor loss, critic loss, advantage computation, entropy bonus) from scratch, you see how the actor uses the critic’s TD error as immediate feedback, verifying the variance reduction claim empirically.

**Stack:**
- **Model:** no pretrained HuggingFace model—define a 2-layer MLP actor-critic (≈64 hidden units each) in PyTorch.
- **Dataset:** OpenAI Gym CartPole-v1 (available through `gymnasium` or `gym`); episodes serve as the training data.
- **Framework:** PyTorch 2.1 with TorchVision 0.16 for utilities; optionally use `gymnasium==0.30` for the environment loop.
- **Compute:** free Google Colab CPU (≤2 min per training run); no GPU needed.

**The recipe:**
1. Install packages: `pip install torch==2.1.0 gymnasium==0.30 wandb` and import `torch`, `torch.nn`, `gymnasium`, `numpy`, and `collections.deque`.
2. Define the actor-critic module: a shared `nn.Sequential` trunk feeding two heads—one softmax actor outputting action probabilities for `CartPole-v1`’s two actions, and one linear critic returning \(V(s)\). Normalize observations with running mean/std from the replay buffer.
3. In each rollout of 5 steps (A2C style), store \((s_t, a_t, r_t, s_{t+1}, \text{done})\), compute \(V(s_t)\) and \(V(s_{t+1})\), form the advantage \(A_t = r_t + \gamma V(s_{t+1}) \cdot (1 - \text{done}) - V(s_t)\), and accumulate actor/critic losses plus an entropy bonus. Use separate Adam optimizers: \(\alpha_\text{actor}=3e-4\), \(\alpha_\text{critic}=1e-3\), \(\gamma=0.99\), entropy weight \(=0.01\).
4. After each rollout, backpropagate the sum of losses—actor loss \(-\log \pi(a_t|s_t) A_t\), critic MSE \(A_t^2\), and entropy—and step both optimizers. Clip gradients to 0.5 for stability. Track episode length and average reward over 100 episodes; expect it to rise toward 200 within ~1500 updates.
5. Evaluate every 100 episodes by running the policy deterministically (choose argmax) for 10 episodes, printing the mean reward. Save the actor-critic state dict when the evaluation reward exceeds 195 and log the loss curves to WandB or TensorBoard.

**Expected outcome:** A checkpointed A2C agent that reliably gets ≥195 average reward on CartPole-v1 and an accompanying plot showing actor loss, critic loss, and TD error magnitude converging.

- **CS student:** Run step 3 with a pure notebook (no WandB) and add a Colab slider to adjust \(\gamma\) and watch how the critic’s TD error changes, proving that the critic’s accuracy determines the actor’s convergence rate.
- **Applied engineer:** Package the trained PyTorch model with TorchScript, quantize the actor head to int8, and serve it via a FastAPI endpoint that responds to JSON state vectors with action probabilities, measuring 10ms latency on an A10.
- **Applied researcher:** Hypothesize that doubling the critic’s depth improves convergence; clone the recipe, add a second hidden layer to the critic only, and run two runs (baseline vs. deeper critic) to compare variance in the advantage estimate over time.
- **Frontier researcher:** Probe the open question on dynamic learning rates by introducing an exponential moving average of the critic TD error to adjust \(\alpha_\text{actor}\) and \(\alpha_\text{critic}\) online, and measure whether the system avoids policy-critic feedback loops identified in ECHO (2025).

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*