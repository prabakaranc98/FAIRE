---
title: Markov Decision Processes
slug: markov-decision-processes
layer: core
subject: 06-reinforcement-learning
page_type: concept
state: drafted
authors_anchored: [sutton, russell, silver, schulman, bello]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [probability-theory, value-iteration, policy-gradients]
tags: [mdp, reinforcement-learning, value-iteration, policy-evaluation, resource-allocation, safety]
updated: 2025-01-04
has_mvb: true
---

# Markov Decision Processes

Imagine a self-driving car approaching a bridge slick with black ice: a naive controller might twist the wheel to stay in its lane without asking what the slip will mean for the next three seconds. An MDP-aware controller instead funnels that steering choice through a probabilistic model of how inertia, friction, and upstream decisions evolve, asking “if I take action A now, what distribution of bends, tilts, and sensor readings will unfold?” That is the point: every sequential decision problem worth automating—from cloud autoscalers to multi-token LLM planners—drives us toward an MDP because actions reshape the range of future states under uncertainty. By the end of this page you should be able to map a real business constraint to state, action, and reward tensors, derive the Bellman recursion that justifies value iteration and policy search, and understand how modern papers layer LLMs and diffusion priors on top of those tensors when transitions themselves are learned.

## The territory

MDPs sit at the intersection of control theory, dynamic programming, and stochastic optimization. You recognize the pattern whenever the task is “choose actions so that some cumulative payoff is good even though the future is uncertain.” Classic robotics formulations—reach the goal while minimizing energy—are MDPs, as are recommender systems that trade immediate clicks for long-term retention. They are the formal structure under every policy gradient, every actor-critic, and every RLHF tuning loop. What distinguishes an MDP is the assumption that the environment response can be summarized by a transition kernel \(P(s' \mid s, a)\) and a reward function \(r(s, a)\), while the decision-maker only observes a state \(s\) before choosing an action \(a\). The consequence is that we no longer optimize per-action greedily; we reason about sequences, expectations, and discounted sums. This perspective is what lets LLM fine-tuning algorithms align token-level rewards with coherent narratives by treating each next-token decision as an action that changes the distribution over future tokens, as demonstrated by Melo et al. (2025) [arxiv:2510.00819](https://arxiv.org/abs/2510.00819). How does any of this actually work under the hood?

## How it works

We begin with the building blocks: the state space \(S\), the action space \(A\), the transition kernel \(P: S \times A \times S \to [0,1]\), and the reward \(r: S \times A \to \mathbb{R}\). The policy \(\pi(a \mid s)\) maps states to distributions over actions, and the key quantity is the state-value function under policy \(\pi\):

\[
v^\pi(s) = \mathbb{E}_{\pi, P}\left[\sum_{t=0}^\infty \gamma^t r(s_t, a_t) \,\middle|\, s_0 = s\right],
\]

where \(s_0\) is the starting state, \(\gamma \in [0,1)\) is the discount factor that ensures the series converges, \(a_t \sim \pi(\cdot \mid s_t)\), and \(s_{t+1} \sim P(\cdot \mid s_t, a_t)\). The expectation is over randomness introduced by \(\pi\) and \(P\). What makes the recursion tractable is the Bellman equation: \(v^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot \mid s)}\left[r(s,a) + \gamma \mathbb{E}_{s' \sim P(\cdot \mid s,a)}[v^\pi(s')]\right]\). The inner expectation propagates rewards forward through the transition kernel, emphasizing that the current value depends on the entire downstream trajectory.

The simplest solution method is value iteration, which repeatedly applies the Bellman optimality operator:

\[
v_{k+1}(s) = \max_{a \in A}\left[r(s,a) + \gamma \sum_{s'} P(s' \mid s,a) v_k(s')\right],
\]

where \(v_k\) is the value after \(k\) iterations, and the max selects the greedily best action given \(v_k\). Every variable is the same as before—\(r\) is immediate reward, \(P\) is transition dynamics, and \(\gamma\) manages the horizon. Because the operator is a contraction mapping when \(\gamma < 1\), value iteration converges geometrically, but that assumes the true \(P\) is known. In production systems we rarely have the exact kernel; we build approximations either from simulators, past logs, or learned transition models.

The connection between logs and transitions becomes critical when we treat language models as dynamic environments. Melo et al. (2025) made this explicit by mapping token generation trajectories to an MDP where the “state” is the partially generated sequence and the “action” is the chosen next token. The stochastic transition kernel arises from the LLM decoder itself, while reward taps into downstream metrics like helpfulness. Stabilization techniques there—such as reward normalization tied to the log probabilities of future tokens—are the same tools used to regularize the Bellman updates when the reward is sparse.

Transition modeling is even more central when the environment is physical and continuous. Flexible locomotion learning with diffusion model predictive control (Anonymous et al. 2025) [arxiv:2510.04234](https://arxiv.org/abs/2510.04234) replaces handcrafted transition kernels with diffusion priors that sample plausible next states conditioned on current proprioceptive observations. In that setup, the action sequence is optimized against samples from the diffusion-based \(P\), and rewards encode stability, energy, and locomotion smoothness. The diffusion sampler itself is trained to match the conditional density \(P(s_{t+1} \mid s_t, a_t)\), so the same machinery that generates images is repurposed to hallucinate next robot poses under different actions.

Because transitions are often learned, robustness becomes critical. Robust counterfactual inference in Markov decision processes (Anonymous et al. 2025) [arxiv:2502.13731](https://arxiv.org/abs/2502.13731) develops non-parametric confidence bands around \(P\) estimated from logs, quantifying how value functions would change under slightly different kernels. The bounding technique produces intervals on future rewards that can feed into conservative policies or safety constraints, which is invaluable when you cannot afford large mismatches between training logs and deployment distribution.

Value iteration gives you the optimal value function, but you still need a policy. The greedy policy derived from \(v^*\) is:

\[
\pi^*(s) \in \arg\max_{a \in A}\left[r(s,a) + \gamma \sum_{s'} P(s' \mid s,a) v^*(s')\right],
\]

where every variable has been explained above. In tabular settings this argmax is cheap. In large or continuous state/action spaces we approximate \(v\) with function approximators and perform policy improvement by gradient ascent. That is where modern algorithms such as PPO enter; ensuring they converge requires controlling the policy updates' step size and the trust region. Anonymous et al. (2026) — An Approximate Ascent Approach To Prove Convergence of PPO [arxiv:2602.03386](https://arxiv.org/html/2602.03386) provides a theoretical scaffold by showing how clipping and adaptive learning rates enforce monotonic improvement bounds, even when the trajectory distribution shifts over time.

The lesson for practitioners is this: even though the Bellman equations assume stationary \(P\), your inference stack often makes \(P\) depend on large, autoregressive models whose parameters shift during training and whose hidden representations (the “state” you condition on) drift. Anonymous et al. (2026) [arxiv:2604.08865](https://www.arxiv.org/pdf/2604.08865) studies policy behavior when such non-stationarity appears by analyzing gradient dynamics in the presence of drifting representations, while another anonymous 2026 paper [arxiv:2602.01156](https://www.arxiv.org/pdf/2602.01156) characterizes what it means for a learned transition model to preserve safety constraints as its latent space evolves. These recent works operate on the same Bellman backbone we described—they just make explicit how the assumptions about \(P\), \(r\), and the state embedding are violated and how to spot that during training.

Working with these violations is precisely why we build MDPs in the wild: the state representation might be a vector of CPU utilization, request queue depth, temperature, and demand forecasts. The action might be a tuple of “scale up/down digital twins, shift workloads, spin up on-demand GPUs.” Reward encodes cost, latency SLAs, and downstream revenue. When this is done correctly the Bellman equation guides your value iteration updates, and value iteration transforms raw domain knowledge into a policy that honors long-term trade-offs even in the face of random perturbations.

## Where the field is now

The current research frontier keeps pushing on the central tension between data efficiency and transition robustness. Melo et al. (2025) [arxiv:2510.00819](https://arxiv.org/abs/2510.00819) stabilizes policy gradients for LLMs by aligning token-level trajectories to conservative estimates of future reward, reducing the number of rollouts needed to tune models for reasoning tasks. In continuous-control domains, Anonymous et al. (2025) [arxiv:2510.04234](https://arxiv.org/abs/2510.04234) demonstrates that diffusion model predictive control can replace handcrafted simulators inside standard MPC loops, allowing locomotion policies to plan over long horizons by sampling latent dynamics that capture high-frequency motion. The new robustness stories, from Anonymous et al. (2025) [arxiv:2502.13731](https://arxiv.org/abs/2502.13731), extend these ideas by computing counterfactual bounds, so when a diffusion prior drifts or when an LLM updates its own hidden state, you still know how much your value estimate could swing. On the optimization side, recent convergence proofs for PPO—most notably Anonymous et al. (2026) [arxiv:2602.03386](https://arxiv.org/html/2602.03386)—confirm that clipped policy gradients behave like approximate ascent under well-chosen step sizes, giving practitioners guidance on how to tune entropy bonuses and KL penalties.

As for engineering, Meta’s decision intelligence platform ReAgent (ai.meta.com/research/reagent/) remains one of the most mature systems. ReAgent integrates offline policy evaluation, online policy deployment, and long-horizon value estimation for ads, ranking, and budget allocation across billions of events per day. Its stack couples MDP modeling with distributed data pipelines that turn historical logs into transition matrices and reward estimates, and the platform exposes a safety layer that checks proposed actions against business constraints before allowing them to run. That kind of engineering frontier is what we aim to mimic in the Build section: you take production-grade signals (CPU, latency, demand) and compress them into a tabular MDP, then apply value iteration to produce a policy that respects cost targets.

Looking ahead, the new papers collectively say: build better transition models, regularize policy updates, and never forget to track what would happen if \(P\) changed. The Value Iteration and PPO stories still anchor the practice while diffusion priors and counterfactual bounds are the newest tools in the kit.

## What's still open

1. How can policy safety be guaranteed when the transition dynamics come from non-stationary, autoregressive LLMs whose internal state representations change during online fine-tuning? Existing 2026 analyses prove convergence only for stationary \(P\), so an open question is how to retroactively correct for representation drift without full retraining.

2. Can diffusion priors be conditioned on business constraints (cost budgets, carbon targets) so that the resulting MDP transition model respects a safety envelope? The current works either sample unconstrained state predictions or hard-clamp the actions post hoc.

3. Is there a practical method to quantify the Bellman error when both rewards and transitions are estimated from heterogeneous logs — for example, mixing structured telemetry (CPU, I/O) with unstructured text (logs, alerts)? Robust counterfactual bounds hint at the answer, but a scalable estimator that avoids worst-case pessimism remains missing.

4. Once a policy is deployed in a cloud environment, can we detect mis-specification of the transition model quickly enough to trigger rollbacks before business KPIs suffer? Model-monitoring frameworks exist for supervised learning but not yet for MDPs at scale.

## Where to read next

If you want to dive deeper into the planning algorithms that translate Bellman equations into code, → [[value-iteration]] walks through the contraction guarantees and grid-world examples that you will be simulating in the MVB. The engineering sibling is → [[policy-gradient]] which shows how PPO and TRPO instantiate the same state-action calculus from a model-free perspective. For the theoretical underpinning of why LLMs can be treated as environments, → [[rlhf]] explains how reward models, policy gradients, and offline data interact to produce stable fine-tuning.

## Build it

The practical question we now resolve is how to turn a business goal—keep a multi-tenant AI service within latency and cost targets—into an MDP policy that can be validated on Colab. By tabulating states, actions, and transitions with NumPy and solving them with value iteration, you will see exactly how reward shaping and transition design control convergence. The following recipe walks you through constructing that ecosystem, just as a production rollout team would do before shipping.

**What you're building:** a NumPy-based autoscaler simulator whose optimal policy balances cost and latency by solving the constructed MDP via value iteration.

**Why this is valuable:** it demonstrates concretely how real-world telemetry is mapped into a transition kernel, how rewards capture SLAs, and how Bellman backups produce action choices that generalize beyond greedy heuristics.

**Stack:**
- **Model:** castorini/mdpr-tied-pft-msmarco (1M+ downloads, model card documents reranking behavior) — used to encode workload patterns via retrieval scores.
- **Model:** castorini/mdpr-tied-pft-msmarco-ft-all (200k+ downloads, fine-tuned on all splits) — serves as a second “policy” signal for cost-sensitive action biases.
- **Dataset:** castorini/msmarco-passage — acts as a stand-in for streaming request logs whose query embeddings correlate with demand spikes.
- **Framework:** Python 3.11 with NumPy 2.1, pandas 2.2, Matplotlib 3.8 for plotting, and HuggingFace `transformers` 4.37 for loading the two MDPR models.
- **Compute:** Free Google Colab (T4, 16GB RAM) — 90 minutes for data download + MDP solving.

**The recipe:**
1. Install and load packages with `pip install numpy pandas matplotlib transformers datasets accelerate` and `from datasets import load_dataset` to fetch `castorini/msmarco-passage`.
2. Preprocess the dataset by grouping queries into minute-long windows, averaging token lengths, and using `castorini/mdpr-tied-pft-msmarco` to compute retrieval logits; each window’s quantized latency and throughput become the MDP state vector.
3. Define actions as “scale up/down CPU” plus “switch to spot instances,” then construct the transition tensor \(P[s,a,s']\) using simple physics informed by retrieval scores: if throughput is high and the retrieval confidence is low, the probability of moving to a high-latency state increases.
4. Run value iteration with \(\gamma=0.95\), a reward that penalizes latency over 200ms (+10 per extra 50ms) and adds a fixed cost for spinning up extra instances, and a convergence threshold on max Bellman error below 0.01 (expect ~120 iterations).
5. Evaluate by simulating 10,000 steps taking the greedy policy \(\pi^*(s)\), reporting average latency (target < 220ms) and daily cost (target < \$120), and plot the value function surface.

**Expected outcome:** a policy table or state-to-action mapping you can export, a plot showing the value surface over latency/cost states, and a short report demonstrating that the solver chooses scale-down when demand is low and scale-up when latency threatens the SLA.

- **CS student:** Run the same pipeline on Colab but skip the second HF model; use a smaller subset of the dataset (10k windows) and use pandas only for preprocessing to keep the experiment under one hour.

- **Applied engineer:** Extend the policy to export decisions via ONNX and measure a simulated p95 latency under a 5-minute sliding window in a Dockerized endpoint, with quantized transitions to mimic production drift.

- **Applied researcher:** Hypothesize that increasing the penalty on reward for cost will delay scaling; test three penalty levels, plot their effect on action frequencies, and report the change in expected latency-cost Pareto frontier.

- **Frontier researcher:** Probe the open question about autoregressive LLM transitions by replacing the deterministic transition tensor with samples from `castorini/mdpr-tied-pft-msmarco-ft-all` conditioned on recent windows, then measure how deviation from the tabular \(P\) affects policy safety bounds from Robust Counterfactual Inference.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*