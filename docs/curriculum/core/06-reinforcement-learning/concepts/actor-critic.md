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

How do you learn a piece of music when the only feedback you get is applause at the end: "good" or "bad"? The REINFORCE-style policy gradient is just this—the actor performs a whole episode, waits for the final return, and then nudges its parameters in proportion to the total reward; each update collapses dozens of stochastic decisions into a single noisy scalar. Try the same trick blindfolded at the piano and you will repeat the same mistakes forever. Actor-critic architectures solve this by introducing a coach who whispers, "You just played better than usual in this bar" or "worse than usual," turning a single episode’s feedback into fine-grained, low-variance guidance. In the process, the critic learns a value estimate that grounds the policy gradient, temporal-difference learning keeps both networks stable, asynchronous rollouts or replay buffers scale the method to millions of timesteps, and, as you will see, a working Advantage Actor-Critic (A2C) agent learns CartPole-v1 in two minutes on free Colab hardware. The difference between blindly chasing rewards and dancing with a critic is the core insight this page delivers.

## The territory

Reinforcement learning sits on a spectrum whose endpoints are familiar: value-based algorithms such as Q-learning treat the policy as the argmax of a learned action-value function and therefore sidestep modeling the policy directly, while pure policy gradient methods weight each action by the return of an entire episode and therefore suffer from high variance that explodes with horizon length. Actor-critic lives between those endpoints. Its actor still updates via gradient ascent on the expected return, but the critic replaces the noisy episode-level return with a learned value function, serving as a baseline that reduces variance without introducing bias. The critic can estimate the state-value \(V^\pi(s)\) that predicts the expected return from state \(s\) under policy \(\pi\), or the action-value \(Q^\pi(s,a)\) that adds dependence on the chosen action \(a\). Either way, the critic is trained with temporal-difference (TD) learning so that it provides localized feedback instead of waiting until the episode ends. Actor-critic was already solving hard control problems before deep networks arrived; the elevator-control system in Crites & Barto’s early work (1995) [Crites & Barto 1995, http://all.cs.umass.edu/pubs/1995_96/crites_b_95.pdf] demonstrated that decoupling policy and value learning stabilizes large, nonstationary systems. Pulling on that thread gives the intuition for the mechanism: the actor performs trial-and-error, the critic measures the deviation from its expectation, and the actor uses that deviation to adjust in the direction that would have made the critic happier. How does that mechanism translate into concrete gradients, objectives, and stability guarantees? The next section walks through it step by step.

## How it works

Actor-critic begins with the policy gradient theorem, which states that the gradient of the expected return \(J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]\) with respect to policy parameters \(\theta\) can be written as

\[
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a \mid s)\, A^{\pi_\theta}(s,a)\right].
\]

where \(d^{\pi_\theta}(s)\) is the discounted state visitation distribution under policy \(\pi_\theta\), \(\pi_\theta(a \mid s)\) is the policy’s conditional probability of action \(a\) in state \(s\), and \(A^{\pi_\theta}(s,a)\) is the advantage—the difference between the quality of taking \(a\) now and the baseline expectation. The advantage \(A^{\pi_\theta}(s,a)\) can be estimated by the critic, which provides the necessary variance reduction compared to using raw returns. The simplest choice is \(A^{\pi_\theta}(s,a) = Q^{\pi_\theta}(s,a) - V^{\pi_\theta}(s)\), so the critic has to approximate either the action-value \(Q\) or the state-value \(V\). In practice, representing \(Q\) or \(V\) with a neural network \(V_\phi(s)\) (with parameters \(\phi\)) permits bootstrapping and allows the actor to use shorter, lower-variance targets than full episode returns.

Training the critic uses TD learning. With one-step TD, we minimize the squared TD error

\[
\mathcal{L}_{\text{critic}}(\phi) = \mathbb{E}_{s,a,s'}\left[\left(r + \gamma V_\phi(s') - V_\phi(s)\right)^2\right].
\]

where \(r\) is the immediate reward observed after transition \(s \rightarrow s'\), \(\gamma \in [0,1)\) is the discount factor, and \(V_\phi\) is the current value estimate. This loss updates the critic towards matching its one-step bootstrap target \(r + \gamma V_\phi(s')\), which is cheaper to compute and has far lower variance than waiting for the full return. The actor, meanwhile, receives the advantage estimate computed from this critic and performs gradient ascent on the policy parameters using the gradient formula above.

### Compatible function approximation and two time-scales

Just learning any critic is not enough: the gradient estimate must remain unbiased. Konda & Tsitsiklis (2000) analyzed compatible function approximation for linear policies, showing that if the critic is constrained to be compatible with the policy parameterization and learns on a faster timescale, the variance of the actor’s update is reduced without introducing bias. The modern deep RL analog relies on two-time-scale stochastic approximation, where the critic uses a smaller learning rate or performs multiple updates per actor step. In practice, the critic update often follows the actor update in the same minibatch to ensure that the value estimate reflects the current policy. The theoretical guarantee of converging to a local optimum emerges when the ratio of learning rates keeps \(\phi\) (critic) ahead of \(\theta\) (actor), as formalized by the two time-scale proofs; practitioners achieve this by simply giving the critic a few gradient steps for every actor step.

The 2012 treatment of the actor-critic policy gradient (ArXiv:1205.4839) further clarifies the decomposition of the gradient into actor and critic components and introduces natural gradient preconditioning that respects the Fisher metric of the policy family. That preconditioning reshapes the actor’s update to move along directions that change the policy distribution uniformly, which again lowers variance and improves stability when using function approximation. Importantly, this theory shows that the critic’s objective can be derived from minimizing the mean-squared projected Bellman error, connecting it to the policy’s Fisher information through the so-called compatible features, which are simply the gradients \(\nabla_\theta \log \pi_\theta(a \mid s)\). This insight is why most modern deep actor-critic implementations keep the gradient flowing through the critic, even though the critic does not directly influence the actor’s architecture.

### Temporal-difference feedback stabilizes the actor

Temporal difference (TD) returns form the backbone of actor-critic updates. In multi-step TD, we define the \(n\)-step return \(G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V_\phi(s_{t+n})\). The advantage estimate then becomes \(A_t = G_t^{(n)} - V_\phi(s_t)\), which uses several future rewards but still bootstraps at \(s_{t+n}\), maintaining lower variance than a full-episode Monte Carlo return. Generalized Advantage Estimation (GAE) (Schulman et al.) extends this idea by exponentially weighting every possible \(n\)-step return with a decay factor \(\lambda\), effectively interpolating between Monte Carlo and TD. When \(\lambda\) approaches 1, GAE recovers high-variance Monte Carlo returns; when \(\lambda=0\), it reduces to one-step TD. Tuning \(\lambda\) and the critic learning rate controls the bias–variance trade-off, and an actor using this advantage receives more informative gradients that align with the critic’s confidence.

### Variance reduction across batches and multiple workers

Batching and parallelism further stabilize actor-critic training. Mnih et al.’s Asynchronous Advantage Actor-Critic (A3C) (Mnih et al. 2016, [arXiv:1602.01783](https://ar5iv.labs.arxiv.org/html/1602.01783)) replaced experience replay with a set of parallel actors each interacting with separate environment instances. Each worker computes gradients independently and periodically updates shared policy and value networks; this decorrelates the data and keeps the policy fresh without storing massive replay buffers. The asynchronous scheme exploits multi-core CPUs, increasing throughput and stability, especially when training on brittle environments where replay causes stale targets. Every worker computes its own advantage estimates \(A_t\) and updates the shared parameters via gradients; the collective effect is similar to large-batch gradient descent while the critic keeps acting as the stable baseline.

Later refinements in 2017 (Schulman et al. 2017, [https://arxiv.org/pdf/1711.04755](https://arxiv.org/pdf/1711.04755)) introduced the clipped surrogate objective that PPO uses. In PPO the critic updates on the same minibatch as the actor, but the actor’s updates are constrained by the ratio

\[
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)},
\]

where \(\theta_{\text{old}}\) is the policy before the update, and the clipped objective uses \(\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\) to prevent large policy shifts; here \(\epsilon\) is a hyperparameter (commonly 0.2). The critic’s TD error is still the driving signal inside \(A_t\), so actor-critic makes PPO feasible: the critic informs how much each action actually improved the expected return, and the clip prevents the actor from trusting noisy advantage estimates too strongly.

### Working with entropy and implicit critics

Recent work on alignment and large language models explores actor-critic variants where the critic is implicit. For example, modern algorithms approximate token-level rewards with policy entropy adjustments rather than an explicit value head, yet they still treat the negative log-likelihood advantage as a critic-guided signal. The critic’s classical role—providing low-variance advantage estimates—remains the same, but now the “value” is derived from learned reward models or entropy terms, not a separate network. Keeping the theoretical guarantees from two-time-scale updates becomes harder in these implicit cases, which is why the open problem at the end of the page focuses on designing critics whose safety guarantees survive non-stationary policy spaces.

### Failure modes and practical fixes

Actor-critic can still fail if the critic becomes too accurate relative to the actor (leading to policy collapse) or too outdated (leading to wrong advantages). Regularizing the critic with weight decay, limiting the target network updates, or using gradient clipping on the value loss prevents divergence. Another failure mode is reward hacking, where the critic learns to reward unhelpful policy shifts because the value estimates drift. Ensuring that the critic itself is evaluated on holdout rollouts or has an auxiliary loss that enforces monotonic improvement on historical data can help. Finally, scaling to pixel-based inputs requires convolutional encoders for both actor and critic, which is why modern implementations share convolutional backbones with separate heads to reduce parameters and ensure aligned representation learning.

## Where the field is now

The contemporary actor-critic landscape has expanded both as a research frontier and an engineering deployment story. On the research side, DreamerV3 (Hafner et al. 2023, [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)) combines world models with actor-critic updates, using imagined rollouts to train both actor and critic so that planning occurs within latent dynamics. The critic evaluates imagined trajectories, enabling the actor to optimize longer horizons without real environment interaction, which cuts sample requirements by orders of magnitude. Another cutting-edge direction is offline reinforcement learning; algorithms like AWAC (Nair et al. 2020) and IQL (Kumar et al. 2020) keep actor-critic structure but constrain policy updates based on the dataset distribution, which is essential when replaying human-demonstrated actions or large-scale logged data.

On the engineering side, actor-critic forms the backbone of almost every deployed RLHF pipeline. PPO, the actor-critic variant from Schulman et al. 2017, is used in OpenAI’s ChatGPT and Anthropic’s Claude training runs to fine-tune language models with human preferences. Both companies use a critic-like reward model to compute token-level advantages, and then update the policy network within trust-region-style constraints, matching the narrative of actor-critic with clipped updates and entropy regularization. Serving such models at scale requires batching, quantized inference, and careful monitoring of value estimates to avoid reward hacking when the critic becomes adversarially exploited; these operational lessons are now standard in industry RL toolchains.

This research + engineering stack shows how actor-critic continues to win the variance–bias trade-off for both academic benchmarks (Dreamer-style planning, offline RL) and real-world alignment problems (RLHF at scale). Yet every deployment reveals vulnerabilities, which we turn to next.

## What's still open

1. How can we design implicit critic architectures that mathematically guarantee immunity to reward hacking in highly expressive, non-stationary policy spaces such as those of large language models? Explicit critics can be regularized or constrained, but implicit critics derived from reward models or entropy terms lack a clear safeguard.

2. Can an actor-critic system provide theoretical convergence guarantees when the critic is trained on imagined rollouts produced by a learned world model, as in Dreamer-style agents, and the imagined distribution drifts away from the real environment?

3. In offline RL, how can actor-critic updates be made conservative enough to respect the dataset distribution while still allowing the actor enough flexibility to improve—especially when the critic is estimated from batch data that may contain spurious correlations?

4. What are the minimal trust-region constraints that maintain stability in large-scale PPO deployments, and how do they interact with adaptive entropy bonuses that change the effective reward landscape every few epochs?

## Where to read next

If you want the theoretical foundation for the variance–bias trade-off, → [Policy gradient](policy-gradient.md) explains the policy gradient theorem and its compatible function approximation proof. The practical implementation of the clipping and trust-region mechanisms that keep modern actor-critic variants stable lives in → [Proximal Policy Optimization](ppo.md). For a wider perspective on value-function learning that feeds the critic, → [[value-functions]] lays out Bellman operators and TD-learning in full detail.

## Build it

Training a working Advantage Actor-Critic agent from scratch forces every reader to juggle a policy network, a value network, TD error computation, and entropy regularization, so it mirrors the core variance–reduction idea in code.

**What you're building:** A PyTorch Advantage Actor-Critic (A2C) agent that reaches an average reward above 475 on Gym’s `CartPole-v1` within two minutes on a free Colab CPU.

**Why this is valuable:** The build exercises the actor’s log-prob gradient, the critic’s TD loss, and the advantage computation in one tight loop, making the critic’s role in stabilizing updates tangible.

**Stack:**
- **Model:** Build the actor and critic as two-head MLPs (no pretrained weights; tied hidden layers and separate heads)
- **Dataset:** Gym’s `CartPole-v1` environment (HuggingFace dataset `gym/cartpole-v1`)
- **Framework:** PyTorch 2.1 + `torchvision` 0.17, `gymnasium==0.28.1`
- **Compute:** Free Colab CPU (dual-core); expect training to finish in ~90 seconds

**The recipe:**
1. Install `pip install torch==2.1.0 gymnasium==0.28.1 numpy matplotlib`, then import `torch`, `torch.nn.functional`, and `gymnasium`.
2. Initialize a shared MLP encoder with two hidden layers (128 units each, ReLU) feeding two heads: a categorical policy over 2 actions with softmax and a scalar value output.
3. Collect rollouts of 5 steps, compute rewards-to-go via discounted sum with \(\gamma=0.99\), and form advantages \(A_t = G_t - V_\phi(s_t)\); update the critic with MSE loss `MSE(V_\phi(s_t), G_t)` and the actor with `-(log_prob * A_t)` plus an entropy bonus coefficient of 0.01.
4. Run 1000 updates (~5000 environment steps), logging the mean episode reward every 10 updates; expect the mean to cross 400 around update 80 and exceed 475 by update 150.
5. What you now have is a PyTorch checkpoint that encodes both actor and critic weights, a training log showing convergence, and a script that can sample CartPole episodes with the learned policy.

**Expected outcome:** A checkpointed attacker-critic model and a training log demonstrating convergence above 475 in CartPole, ready to plug into later arcs (e.g., PPO step).

- **CS student:** Run the same recipe on an RTX 4070 but reduce episode length to 20 steps during debugging; once the debug run is stable, raise it back to 200 to match CartPole’s horizon.
- **Applied engineer:** Export the trained policy head to ONNX, quantize to INT8 with `torch.quantization.quantize_dynamic`, and serve through a simple Flask endpoint that benchmarks p50 inference latency (<15 ms) on a CPU with the critic weights frozen.
- **Applied researcher:** Add an ablation where you disable the entropy bonus and plot the resulting average advantage magnitude; hypothesize that the entropy stabilizer reduces actor variance by at least 15%.
- **Frontier researcher:** Probe the open question on implicit critics by replacing the explicit value head with a learned reward model trained on human demonstrations and measuring whether PPO’s trust-region clipping still prevents reward hacking under distribution shift.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*