---
title: Policy gradient
slug: policy-gradient
layer: core
subject: 06-reinforcement-learning
page_type: concept
state: drafted
authors_anchored: [sutton, barto, schulman, silver]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [value-functions, actor-critic, llm-alignment]
tags: [policy-gradient, llm-alignment, advantage-estimation, grpo, rlhf]
updated: 2025-01-15
has_mvb: true
---

# Policy gradient

Imagine grading a ten-step math proof by handing the student a single pass/fail slip at the end. The only feedback the student receives is whether the whole chain of inference succeeded, not which step was inconsistent or how they could change their next line. That is the reality classic policy-gradient algorithms face when they work with long chains of token generation in LLMs: every token is credited by the final reward signal, which makes variance skyrocket and training fragile unless elaborate critics or baselines are introduced. As the frontier shifts to instruct-tuned and RL-aligned models, the question becomes not “Can we estimate the gradient of the cumulative reward?” but “Can we deliver relative, token-level feedback that keeps the signal local enough to tinker with the policy every few tokens?” This page traces that shift, explains how modern policy gradients trade critics and trajectory-level advantages for token-wise estimates and KL regularization, and leaves you ready to train a GRPO loop on a free Colab instance so you can feel this stability difference in your own project.

## The territory

Policy gradient lies at the heart of reinforcement learning because it directly optimizes the parameters of a stochastic policy using the gradient of expected return. In bandit-style settings the gradient is tractable, but once actions become sequences—like the stream of tokens an LLM emits—the variance of the log-likelihood gradients overwhelms training unless some form of credit assignment is introduced. Traditionally, the RL canon responded with critic networks: estimate the value of a trajectory, subtract that baseline from the return, and hope the reduced variance outweighs the bias. That dependency is acceptable in simulated robotics when thousands of episodes can be collected, but RLHF and instruction tuning do not afford such luxury; human preferences are expensive and sequences run into dozens of steps. The territory this page covers is the modern variant of policy gradient that abandoned monolithic critics in favor of relative, token-level advantage updates regularized by KL constraints—what the field now calls critic-free policy gradients. It borrows from classic ascent analysis but reinterprets every move through the lens of LLM alignment, where policy drift, token entropy, and KL rollback control become the levers that keep reasoning stable. How does this actually work? The next section digs into the math of token-level averages, entropy-aware rewards, and the group-relative policy optimization (GRPO) algorithms people now deploy on large language models.

## How it works

The starting point remains the fundamental policy gradient theorem: the update proceeds in the direction that increases the log probability of actions that led to higher-than-expected returns. Formally, one writes the optimization objective as

\[
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t) A_t^\pi \right]
\]

where \(\pi_\theta\) is the policy parameterized by \(\theta\), \(\tau\) denotes a trajectory \((s_1, a_1, \dots, s_T, a_T)\) sampled from the current policy, \(A_t^\pi\) is the advantage at timestep \(t\) under policy \(\pi\), \(s_t\) is the current state (in the case of LLMs the prefix of generated tokens), and \(a_t\) is the next token. The advantage \(A_t^\pi\) measures how much better the taken action was compared to the expected value of the state. Classic implementations estimated it via a critic network and Monte Carlo rollouts, but in long-horizon token generation the variance of a single scalar advantage per prefix swallows the training signal. With critic-free gradients, we instead build a relative advantage that lives at the token level.

### Token-level relative advantage

Token-level relative advantage replaces the trajectory-wide return with a reward that reflects the immediate impact of a token relative to a baseline derived from other tokens in the same context. We define \(\delta_t = r_t - \bar{r}_t\) where \(r_t\) is the reward associated with the token generated at step \(t\) (for example, the change in preference score attributed to inserting or omitting that token) and \(\bar{r}_t\) is a local baseline constructed from tokens that share a similar prefix or token identity. The update becomes

\[
\nabla_\theta J_{\text{rel}}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t) \delta_t \right]
\]

where the baseline \(\bar{r}_t\) shrinks the variance by canceling out the broad-scale reward that every token experiences equally, leaving only the relative advantage that is indicative of the specific token choice. The critical difference for LLMs is that computing \(\delta_t\) does not require a critic network: it leverages group statistics or adaptive entropy baselines computed from the same generation batch, avoiding the instability of training a separate value head.

This formulation is the foundation of Group Relative Policy Optimization (GRPO). GRPO gathers tokens into groups (e.g., tokens sharing the same prompt or generated under the same temperature), computes the group mean reward, and assigns advantages relative to that mean. The policy gradient then nudges the model to increase the likelihood of tokens that performed better than their group mates while keeping the overall distribution stable.

### KL regularization and entropy rewards

Stability also demands explicit controls on how far the policy drifts from its initialization, especially when the reward signal is sparse. KL regularization plays that role. We augment the objective with both forward and reverse KL terms, leading to the KL-regularized RPG formulation in Yu et al. (2025) [arxiv:2505.17508](https://arxiv.org/abs/2505.17508). The augmented objective is

\[
\mathcal{L}_{\text{KL}}(\theta) = -\mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{t=1}^T \log \pi_\theta(a_t \mid s_t) \delta_t \right] + \lambda_{\text{F}} \text{KL}\left(\pi_\theta \,\|\, \pi_{\text{ref}}\right) + \lambda_{\text{R}} \text{KL}\left(\pi_{\text{ref}} \,\|\, \pi_\theta\right)
\]

where \(\pi_{\text{ref}}\) is a frozen reference model (often the pretrained LM), \(\lambda_{\text{F}}\) and \(\lambda_{\text{R}}\) are coefficients for the forward and reverse KL, and \(\delta_t\) is the token-level relative advantage. The forward KL term prevents the updated policy from straying too far, while the reverse KL encourages the policy to keep its mass concentrated on high-likelihood sequences, thus preserving coherence. The symmetry ensures that even without a critic, the policy neither collapses to deterministic degeneracy nor drifts into low-quality, high-reward regions. 

To provide a smoother reward for individual tokens, GTPO/GRPO-S (arXiv:2508.04349) reinterprets policy entropy as a dynamic reward signal: higher entropy indicates that the model is still exploring, so the reward encourages tokens that maintain an optimal level of uncertainty, preventing overconfidence in the face of noisy preferences. The token-level reward becomes

\[
r_t = \alpha \cdot \text{pref}_t + \beta \cdot \mathcal{H}\left(\pi_\theta(\cdot \mid s_t)\right)
\]

where \(\alpha\) scales the preference-based reward \(\text{pref}_t\) and \(\beta\) scales the entropy \(\mathcal{H}\). The policy gradient remains relative, but the reward now tracks both preference signal and entropy, allowing the model to allocate exploration pressure adaptively.

### Critic-free convergence with offline value separation

One lingering concern is convergence without a critic. Gao et al. (2025) introduced the \(A^\star\)-PO architecture where offline value estimation is entirely separated from online policy updates, showing that the policy gradient can converge by minimizing the KL divergence between the current policy and the distribution implied by the offline value function. The offline estimator only needs to provide a high-quality ordering of trajectories, not exact values, so it can be trained using rewards gathered from human preferences or synthetic evaluators. The online policy, using the token-level relative advantages described above, simply performs regularized gradient steps to match the offline ranking. This two-phase separation drastically reduces the compute needed for high-variance critic training because the offline estimator can be trained less frequently and on smaller data budgets.

Mathematically, the convergence argument around \(A^\star\)-PO leverages the policy update 

\[
\theta_{k+1} = \arg\min_\theta \text{KL}\left(\pi_\theta \,\|\, \pi_{\text{offline}}^{(k)} \right)
\]

where \(\pi_{\text{offline}}^{(k)}\) encapsulates the trajectory ranking produced by the offline value function at iteration \(k\). The token-level gradient acts as a stochastic gradient oracle for this KL minimization, and as the offline estimator improves, the policy follows suit. Importantly, the gradients only ever require the relative ordering, which is what GRPO provides.

### Convergence safeguards

Even with these mechanisms, we still need theoretical assurances, particularly for algorithms derived from PPO. “An Approximate Ascent Approach to Prove Convergence of PPO” (2026) [arxiv:2602.03386](https://arxiv.org/html/2602.03386) introduces an ascent lemma that bounds the deviation between successive policies under clipped objectives. The same analysis carries over to RLHF-style updates: the token-level advantage can be treated as a clipped surrogate objective where \(\delta_t\) is bounded, and the KL regularizers act as the trust-region constraint. The lemma shows that as long as the step size keeps the KL divergence below a threshold and the token-level reward is well-behaved, the expected return improves. This is the theoretical backbone that lets us train critic-free policy gradients with the same confidence as PPO but without the critic.

Practical training still demands careful scheduling of \(\lambda_{\text{F}}\), \(\lambda_{\text{R}}\), entropy weights, and group sizes. Most teams use a small number of tokens per group (e.g., those sharing the same prompt batch) and compute moving averages of their preference (or synthetic reward) to derive the relative advantage. The gradient update then becomes a standard PyTorch backward pass, but the pre- and post-processing steps—advantage normalization, KL penalty tuning, entropy reward scheduling—are what keep training stable.

### Token timing and inference

One final mechanism worth highlighting is how these updates operate during inference. Instead of gathering a whole trajectory, the training loop collects partial sequences (few tokens) and recomputes the relative advantage after each batch. This permits “token-level” updates where only the last few generated tokens require a gradient step, enabling asynchronous parallel updates even when tokens are generated sequentially on a single device.

The combination of relative advantages, KL penalties, and entropy incentives is what allows “critic-free” policy gradient training to work in practice. The next section surveys how these components appear in current research and engineering deployments.

## Where the field is now

Researchers are already coalescing around these ideas. The GRPO-S variant from arXiv:2508.04349 pairs token-level entropy rewards with dynamic KL clamps to produce models that learn more gracefully from sparse human preference data. Their benchmarks on synthetic reasoning tasks—Countdown, Climb, and synthetic math proofs—show that GRPO-S achieves higher reward per token than PPO baselines with critics. On the theoretical front, the ascent proof for PPO from arXiv:2602.03386 provides the convergence certificate that let teams replace critics without risking divergence.

The anonymous 2603.21621 preprint introduces a token-level advantage regularizer that explicitly penalizes deviations from the reference policy’s token distribution, effectively replicating a per-token trust region. Combined with the KL-regularized RPG derivations of Yu et al. (2025), these tools let practitioners maintain tight control over drift while still allowing the model to explore beneficial but rare reasoning paths.

On the engineering side, Meta’s “Understanding Reinforcement Learning for Model Training and Future Directions with GRAPE” article documents how teams are building tooling to support these token-level gradients in large-scale alignment stacks. The Meta system pipelines preference data into a GRPO-style optimizer, monitors per-token KL drift, and orchestrates large language models that are fine-tuned end-to-end on billions of tokens. NVIDIA’s engineering blog on NeMo-RL and Megatron-Core reports production RL training across thousands of GPUs with PPO/GRPO — the blog details how tensor and pipeline parallelism keep tokens-per-second throughput high while enforcing the KL/entropy constraints that keep policy gradients stable. These deployments illustrate an emerging engineering frontier: scaling critic-free, token-level policy gradients to commercial LLMs while the theoretical frontier chases convergence guarantees for these new loss functions. 

The anonymous 2604.08865 preprint explores using relative token advantages to fine-tune policies for dialogue style, while 2602.01156 adds another convergence perspective by showing how repeated relative updates behave like mirror descent with a Bregman divergence tied to the KL penalty. Together, they sketch a research frontier where theory and practice finally agree: relative, token-level policy gradients can be both efficient and provably convergent.

## What's still open

Can we mathematically guarantee convergence of token-level relative advantage updates when applied to diffusion-based language models, where generation occurs in parallel and repeated refinement rather than strict autoregressive sequencing? The standard KL-based proofs rely on the chain rule for autoregressive log-likelihoods, so new techniques are needed for dLLMs.

Can we design a unified regularizer that controls both forward and reverse KL drift adaptively based on reward sparsity, removing the need for manual tuning of \(\lambda_{\text{F}}\) and \(\lambda_{\text{R}}\) while still providing the ascent guarantee of PPO? Evidence so far is heuristic, and a principled derivation would guide automatic scheduling.

How do token-level relative advantages generalize to multi-agent preference settings, where rewards are noisy and conflicting? Existing GRPO variants assume a single reward oracle per batch; extending relative feedback to multiple, possibly contradictory reviewers is still unsolved.

## Where to read next

If you want to see how these ideas relate to traditional actor-critic dynamics, → [Actor-Critic](actor-critic.md) explains how baseline functions shape advantages and why critic avoidance is so disruptive. To understand the particular challenges of aligning LLMs via human feedback, → [[rlhf]] documents the data pipelines and reward models that feed into today’s policy gradients. For the next conceptual shift beyond sequential tokens, → [[diffusion-llms]] explores how parallel generation invites new regularizers and sampling procedures that will soon demand their own gradient estimates.

## Build it

This build proves that a critic-free, token-level GRPO loop can be implemented on a free Colab instance and still stabilize training on a synthetic reasoning game, demonstrating how relative advantages and KL penalties work in practice.

**What you're building:** a bare-bones GRPO training loop that fine-tunes `qwen-2.5-1.5b-instruct` on a synthetic Countdown math task using token-level relative advantages and KL regularization.  
**Why this is valuable:** you will touch every part of the concept—reward shaping, relative advantage normalization, entropy monitoring, and KL rollback—while observing how each term affects loss stability.  
**Stack:**
- **Model:** [Qwen-2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen-2.5-1.5B-Instruct) — 15k+ downloads, instruction-tuned checkpoint with RLHF-ready tokenizer.
- **Dataset:** [math-synthetic/countdown](https://huggingface.co/datasets/math-synthetic/countdown) (create the Countdown task locally by generating random math operations constrained to reach a target within 6 steps).
- **Framework:** PyTorch 2.1 + Hugging Face `transformers` 4.45 + `accelerate` 0.25 for distributed-friendly training.
- **Compute:** Colab T4 (16GB VRAM) with mixed precision, ~3 hours for full run.

**The recipe:**
1. Install `pip install torch torchvision accelerate transformers datasets` and load `QwenForCausalLM` with `trust_remote_code=True`, enabling past-key-value caching for fast sampling.
2. Build the Countdown dataset by generating random integer sets and measurement tokens, tokenizing both the prompt (“Reach 43 using 6 steps: …”) and the model completion; store per-token metadata for rewards computed by a synthetic validator that scores partial solutions.
3. During training, compute the token-level reward \(r_t\) as the difference between the validator score before and after generating token \(a_t\), subtract the group mean baseline per prompt batch to create \(\delta_t\), and clip \(\delta_t\) to \([-1, +1]\) for stability before calculating \(\nabla_\theta \log \pi_\theta(a_t \mid s_t)\).
4. Add KL penalties after each forward pass: compute the forward KL \(\text{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})\) and reverse KL \(\text{KL}(\pi_{\text{ref}} \,\|\, \pi_\theta)\), scale them by \(\lambda_{\text{F}}=0.02\) and \(\lambda_{\text{R}}=0.01\), and backpropagate the sum with the token-level advantage loss.
5. Evaluate by measuring the average token advantage and the synthetic validator reward per prompt; expect the relative advantage loss to stabilize below 0.2 and cumulative reward to approach the threshold set by perfect Countdown solutions.

**Expected outcome:** a checkpoint that can reproduce the synthetic Countdown solutions with high reward per token while maintaining KL drift <0.05, demonstrating the benefit of token-level GRPO.

- **CS student:** Run the same recipe on an RTX 4070 by reducing sequence length to 128 tokens and using gradient checkpointing; this keeps training fast while still collecting token-level relative advantage statistics.
- **Applied engineer:** Package the resulting model behind a vLLM endpoint, quantize it with bitsandbytes 8-bit, and ensure the KL penalties keep inference coherence at p50 latency < 600 ms on an L4 instance.
- **Applied researcher:** Treat the entropy reward coefficient \(\beta\) as a hypothesis variable—test whether doubling \(\beta\) while keeping \(\lambda\) fixed leads to better reaching rates on more complex Countdown instances (>= 8 steps).
- **Frontier researcher:** Probe the open question about non-autoregressive diffusion LLMs by rephrasing the reward signal as the difference between start and end states after a diffusion denoising pass and measuring whether the relative advantage updates still converge.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*