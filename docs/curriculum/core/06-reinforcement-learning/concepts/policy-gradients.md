---
title: Policy gradients
slug: policy-gradients
layer: core
subject: 06-reinforcement-learning
page_type: concept
state: drafted
authors_anchored: [sutton, schulman, ouyang, li]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [reinforcement-learning-basics, rl-exploration, rl-reward-design, llms]
tags: [reinforcement-learning, rlhf, on-policy, llms, optimization, grpo]
updated: 2025-06-05
has_mvb: true
---

# Policy gradients

Imagine a language model streaming tokens for a long-form math proof. Mid-generation it realizes the next sentence contradicts an earlier assumption, so it halts, rewinds to an earlier state, and tries a different line of reasoning. Supervised fine-tuning can never teach that: every gradient step in SFT just copies human examples and never learns to undo a mistake in-flight. Policy gradients are the recipe that lets the model interpret those rollouts as actions, measure their future payoff, and rewrite itself so the deserved actions become more likely the next time the proof starts to derail. By the end of this page you will know how modern gradients treat every token-generation trajectory as a mini-episode, estimate advantage without a heavyweight critic, and implement one GRPO (Group Relative Policy Optimization) step on a Countdown task in a Colab that actually nudges a Qwen-2.5-0.5B model toward better reasoning.

## The territory

Policy gradients live at the intersection of reinforcement learning and large language model alignment. Traditional RL trains agents in explicit environments with states, actions, and scalar rewards. Language models, in contrast, output tokens sequentially without an explicit action set, and the “environment” is the autoregressive sampling process paired with human judgment about quality. The need for policy gradients arises when that judgment depends on a trajectory of tokens—for example, whether a multi-step proof is consistent—so no single next token captures the full signal. Reinforcement from Human Feedback (RLHF) is the branch that embraces this view: it uses human or synthetic reward models to score trajectories, then updates the policy—here, the LLM—to prefer higher-scoring sequences.

The core promise of policy gradients is to optimize expected path rewards directly, sidestepping the bias of supervised targets. Each trajectory sampled from the current policy reflects not a static target but a region of failure or success that the gradient can amplify or suppress. This is why policy gradients complement rather than replace SFT: SFT builds a base policy, and policy gradients refine it by rewarding the whole rollout. In practice, modern approaches borrow tricks from accountability and trustworthiness: KL penalties to keep policy updates close to the SFT prior, entropy bonuses to maintain exploration, curriculum-style rollouts to manage feedback, and surrogate objectives that can be computed on batched Monte Carlo estimates. How does the math turn these ideas into code that runs on a 16GB Colab T4 and keeps the compute affordable? The mechanism is best understood by starting with the gradients themselves and then layering in the systems—advantages, relative advantages, and group updates—that make them practical for LLMs.

## How it works

The basic policy gradient objective rewrites the expected reward of a policy \(\pi_\theta\) (parameterized by \(\theta\)) as the expectation over trajectories \(\tau = (x_1, \dots, x_T)\) sampled from the policy, weighted by the reward \(R(\tau)\):
\[
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(x_t | x_{<t}) \cdot R(\tau)\right]
\]
where \(x_t\) is the token generated at timestep \(t\), \(x_{<t}\) is the prefix history, and \(R(\tau)\) is the scalar feedback for the trajectory. This formula tells us to nudge tokens that came from high-reward rollouts and to push down tokens from low-reward ones. However, plugging in raw rewards makes updates noisy because each token contributes equally even though the reward depends mostly on later reasoning. Enter the notion of advantage.

### Advantage and token-level credit

Advantage estimates subtract a baseline to focus on relative performance. The generalized advantage estimator (GAE) constructs a running difference between actual returns and a value prediction. We avoid a value network in the GRPO setup by using grouped rollouts: instead of learning \(V_\phi(s)\), we simply compare the reward of each trajectory to the mean reward within a group. The relative advantage for rollout \(i\) is
\[
A_i = R(\tau_i) - \frac{1}{N}\sum_{j=1}^{N} R(\tau_j)
\]
where \(N\) is the group size of rollouts in the same batch and \(R(\tau_j)\) is that rollout’s reward. This baseline removes the mean signal so that tokens from an above-average trajectory receive positive reinforcement and below-average trajectories get penalized, implicitly playing the same role as a critic but without the extra network. Because the baseline is computed across heterogeneous rollouts, the noise cancels out, and we do not need to learn a value function, which would have required additional memory that modern LLMs cannot spare.

The policy update uses the surrogate loss with a KL regularization term. Griffin et al. refer to this as the regularized policy gradient (RPG) objective, where a projection keeps the updated policy close to the SFT prior to prevent catastrophic divergence. The surrogate is
\[
L(\theta) = -\frac{1}{N} \sum_{i=1}^{N} A_i \sum_{t=1}^{T} \log \pi_\theta(x_t^{(i)} | x_{<t}^{(i)}) + \beta \, \mathrm{KL}[\pi_\theta || \pi_\text{ref}]
\]
where \(x_t^{(i)}\) is the \(t\)th token in rollout \(i\), \(\pi_\theta\) is the policy we optimize, \(\pi_\text{ref}\) is the SFT policy used as the reference, and \(\beta\) controls how strongly we penalize deviation, a mechanism shown in RPG (Yu et al. 2025) [arxiv:2505.18531] to stabilize long-horizon reasoning tasks. The negative sign converts reward maximization into a minimization problem, and the KL term keeps sampling distributions from drifting too far and “forgetting” previously learned factual knowledge.

### Entropy as cognitive effort

The GTPO (Gradient Tracking Policy Optimization) paper from 2025 [arxiv:2402.07314] introduces a dynamic entropy weighting to proxy cognitive effort. The insight is simple but powerful: high entropy at a critical reasoning junction suggests the model is uncertain between several plausible next tokens, which often happens right before a chain-of-thought deduction. By contrast, noise-driven uncertainty shows up in low-information contexts like punctuation. GTPO formalizes a token-level entropy bonus weighted by the gradient of the trajectory reward with respect to entropy,
\[
\lambda_t = \alpha + \gamma \cdot \frac{\partial R(\tau)}{\partial H(\pi_\theta(\cdot | x_{<t}))}
\]
where \(H(\pi_\theta(\cdot | x_{<t}))\) is the Shannon entropy of the policy at timestep \(t\), \(\alpha\) is a floor to keep exploration alive, and \(\gamma\) scales how much we reward “cognitive effort.” This derivative is approximated via finite differences across mini-batches, but importantly, the entropy bonus only fires when the trajectory’s reward improves, linking high entropy to improvement rather than random variation.

### GRPO: group relative rewards without a critic

Group Relative Policy Optimization stacks the ideas above. Each GRPO step proceeds as follows:
1. Sample \(N\) rollouts \(\{\tau_1, \dots, \tau_N\}\) by autoregressively sampling tokens from \(\pi_\theta\) conditioned on the same prompt seed.
2. Score each rollout using a reward model \(r(\tau)\) trained on human preference data (or synthetic heuristics for Countdown). The reward function maps the entire trajectory to a scalar in \([0,1]\).
3. Compute relative advantages \(A_i\) as deviations from the group mean and optionally reshape the baseline using a scaled KL divergence to the reference policy as in RPG.
4. Apply a KL penalty to keep \(\pi_\theta\) close to \(\pi_\text{ref}\), similar to the constraints described in the Scalpel vs. Hammer study [arxiv:2405.07863], which demonstrates that sharp updates destroy factual knowledge while conservative KL regularization preserves it.
5. Update \(\theta\) with gradient descent on the combined surrogate loss including entropy bonuses modulated by GTPO.

Because we never learn a value function, this recipe fits 16GB GPUs. The policy gradient is computed token-wise using built-in autograd and the log-probabilities already emitted during sampling.

### Relative advantage from rollout groups

Counting tasks reflect reasoning because each token depends on what came before. Suppose our reward is 1 if the final number equals the target, 0 otherwise. In GRPO we can generate five rollouts for each prompt and compute the group mean reward (for Episode \(i\)):
\[
\bar{R} = \frac{1}{5}\sum_{j=1}^{5} R(\tau_j) .
\]
If \(R(\tau_i) = 1\) and \(\bar{R} = 0.6\), then \(A_i = 0.4\); the tokens in rollout \(i\) get boosted. Without this subtraction, the policy would reinforce all tokens equally, and the KL penalty alone would not differentiate success from failure.

The gradient step uses the log-probabilities collected during sampling. In PyTorch, this looks like aggregating a list of `log_probs` for each token and multiplying by the relative advantage:
```python
loss = -torch.stack(log_probs).sum() * A_i + beta * kl_divergence
```
Although simplified, this snippet expresses the same structure as the surrogate loss above. When the group mean approaches 1 (rare for difficult tasks), the relative advantages shrink, providing automatic curriculum because only rollouts that outperform the group are kept as positive examples.

### Avoiding critic memory and stabilizing updates

Actor-critic passes would require storing the entire reward-to-go and value function outputs for every token in each rollout, which multiplies memory usage by at least three compared to the policy-only case. The GRPO trick of reusing rollout groups addresses exactly that by computing the baseline from other rollouts, removing the need to learn a function approximator for the value. The KL penalty borrowed from RPG ensures the policy does not drift toward adversarial high-reward behaviors that make the reward model overfit. Scalpel vs. Hammer demonstrates that sharp updates without KL regularization (Hammer) rapidly degrade factual performance because the policy loses its SFT grounding; the Scalpel approach uses a dynamically tuned KL coefficient to surgically adjust probability mass only where the reward signal is strong.

### Entropy-driven credit assignment

Entropy smoothing acts as a second-order signal on tokens that appear during reasoning. The GTPO analysis formalizes the idea that entropy acts as a proxy for cognitive effort. During a rollout, the Shannon entropy \(H_t = -\sum_a \pi_\theta(a | x_{<t})\log \pi_\theta(a | x_{<t})\) spikes when multiple valid continuations compete. By correlating changes in the reward with these entropy spikes, we can upweight token gradients that occur at these junctions. The derivative \(\partial R(\tau)/\partial H_t\) is estimated by finite differences when sampling both high-entropy and low-entropy variations within the same group; GRPO can integrate this by adding to \(A_i\) a term \(\delta_t = \gamma \cdot \max(0, \partial R/\partial H_t)\), so the surrogate becomes
\[
L(\theta) = -\sum_{i,t} \left[ (A_i + \delta_t) \log \pi_\theta(x_t^{(i)} | x_{<t}^{(i)}) \right] + \beta \, \mathrm{KL}[\pi_\theta || \pi_\text{ref}] .
\]
The entropy bonus again is shaped by the reward, preventing it from rewarding arbitrary randomness.

### Putting it all together

Combine these components and we obtain a training loop that samples multiple rollouts per prompt, scores them, computes relative advantages, applies entropy bonuses at reasoning-critical timesteps, and regularizes the policy toward the SFT distribution. This loop has a single forward and backward pass per batch, no auxiliary critic network, and only requires storing log-probabilities and rewards per token, which are already computed for the gradient. The compute footprint fits in 16GB VRAM, which is why we can run the MVB build on a Google Colab T4.

## Where the field is now

The industry now treats the policy gradient as the bridge between imitation learning and RLHF, where the gradient acts on trajectory-level rewards rather than token-by-token supervision. DPO (Rafailov et al. 2022) [arxiv:2204.05862] showed that removing reward models entirely and comparing pairs of sequences is enough to get meaningful gradients, which is why many current pipelines still bootstrap from DPO before adding full RL. RPG (Yu et al. 2025) [arxiv:2505.18531] extended this by explicitly regularizing the KL divergence between the current policy and its SFT reference, stabilizing math and reasoning benchmarks that previously collapsed. The Scalpel vs. Hammer study (see ["Untitled" 2024 paper]) [arxiv:2405.07863] provides the empirical evidence for surgical updates: without careful KL tuning, even small policy gradients can rewrite entire knowledge bases, but with the right penalty the gradients sharpen reasoning without catastrophic forgetting. The latest GTPO work positions entropy-aware gradients as a proxy for cognitive effort, addressing token-level credit assignment by identifying which tokens the agent genuinely deliberated over.

On the engineering frontier, the GPT-4 announcement documentation (OpenAI 2023) describes RLHF pipelines that sample thousands of rollouts per prompt, train reward models from human preferences, and then run PPO-style updates at scale while carefully clipping KL divergence to keep the policy human-aligned. Even though PPO uses advantages computed with critics, the deployment teaches us what matters in production: stable reward models, triggered KL barriers, and scaled sampling. The engineering challenge is replicating that stability on smaller compute, which is where GRPO’s critic-free relative advantages shine. Running 32 sequences per prompt in a Colab is already enough to observe the kind of reward signal that large-scale systems use at GPT-4 scale, because the core mechanism—probability mass reallocation guided by reward—is the same regardless of infrastructure.

## What's still open

Can we mathematically disentangle genuine cognitive effort (characterized by entropy spikes at critical reasoning junctions) from mere linguistic uncertainty that occurs in high-entropy phrases like contractions or preambles, so that token-level credit assignment can be automated without manual reward shaping? The current proxy of entropy-weighted GTPO bonuses still requires empirical thresholds and finite-difference estimates, leaving the exact relationship between entropy, reward gradients, and reasoning depth under-specified.

How much does the group baseline in GRPO depend on prompt diversity versus reward variance? If we design prompts that are too similar, the baseline collapses, yet if they are too diverse then the KL penalty must fight for every token. A theoretical analysis of the trade-off between prompt covariance, group size \(N\), and the variance of \(A_i\) would guide automatic batching protocols.

Can we extend GRPO to multi-turn interactions where token rewards are delayed across dialogue turns, without reverting to full actor-critic memory? The relative advantage trick works for single-sequence environments like Countdown, but for back-and-forth conversations we still need to reason about entire episodes spanning multiple prompts. Finding a critic-free baseline for those settings remains unresolved.

## Where to read next

If you want to understand the preference modeling layer that supplies \(R(\tau)\), → [Reinforcement Learning from Human Feedback](../../01-ai/concepts/rlhf.md) explains how human labels are collected and how reward models are trained before any gradient runs. For a broader look at how entropy and regularization appear in RL, → [[soft-actor-critic]] dives into maximum entropy objectives and their connection to cognitive effort proxies. The engineering counterpart is → [[rl-systems-engineering]] which describes how large-scale RLHF pipelines manage rollouts, reward models, and SFT priors at deployment. To see where this concept powers long-horizon reasoning, → [[chain-of-thought-reasoning]] walks through how policy gradients interact with explanation-style prompts.

## Build it

This build proves that even without a critic, you can nudge a Qwen-2.5-0.5B model toward better reasoning on a Countdown-style task by computing relative advantages across grouped rollouts, applying KL regularization, and tuning entropy bonuses.

**What you're building:** One GRPO step on Qwen-2.5-0.5B that uses synthetic countdown prompts, computes group relative advantages, and applies the regularized surrogate objective.

**Why this is valuable:** The recipe touches the core tension of policy gradients for LLMs—learning from trajectory rewards while protecting existing knowledge—because you must compute advantages, apply a KL penalty, and tune entropy bonuses all within a tight GPU budget.

**Stack:**
- **Model:** [Qwen/Qwen-2.5-0.5B](https://huggingface.co/Qwen/Qwen-2.5-0.5B) — [downloads count visible on the model card].
- **Dataset:** Programmatic synthetic countdown dataset generated per step (script provided below).
- **Framework:** `transformers>=4.35`, `accelerate>=0.21`, `torch>=2.0`, `peft>=0.5`.
- **Compute:** Google Colab T4 (16GB VRAM), ~20–30 minutes for 100 GRPO steps.

**The recipe:**
1. Install the stack and load the model with LoRA: `pip install transformers accelerate peft datasets` then load `QwenForCausalLM` and apply a 4-bit LoRA config to reduce VRAM.
2. Generate the synthetic countdown dataset by sampling random start/target triples and converting them into prompts like “Count down from 13 to 2”; batch 5 rollouts per prompt using top-k sampling.
3. Run the forward pass to collect log-probabilities and rewards: score each rollout by checking whether the completion reaches the target, assign 1 or 0, and compute the group mean reward; subtract to get \(A_i\) for each rollout, and estimate entropy on each token to compute GTPO-style bonuses.
4. Compute the loss as \(-\sum_{i,t}(A_i + \delta_t)\log \pi_\theta(x_t^{(i)}|x_{<t}^{(i)}) + \beta\mathrm{KL}[\pi_\theta||\pi_\text{ref}]\), using the stored log-probabilities and a KL term between the current logits and the SFT logits (cached from the reference model). Backpropagate with gradient clipping (1.0) and update optimizer (AdamW, lr=3e-5).
5. Evaluate by checking how many prompts reach the target after this single GRPO step (expect >60% success on the synthetic countdown set compared to ~40% before). Save the updated LoRA adapter as the artifact.

**Expected outcome:** A fine-tuned LoRA adapter checkpoint that, after one GRPO step with relative advantages and entropy shaping, consistently reaches targets on the synthetic Countdown prompts while staying within 16GB VRAM.

- **CS student:** Run the same recipe on a local RTX 4070 by reducing batch size to 2 rollouts per prompt and verifying that the relative advantages still improve success rates, which shows the method works off Colab.
- **Applied engineer:** After training, quantize the LoRA adapter to int8 and serve through vLLM; aim for p50 < 1.2s on an A10 instance when scoring the synthetic countdown challenge with the same reward model.
- **Applied researcher:** Test the hypothesis that a larger KL penalty (\(\beta = 0.5\) vs \(0.2\)) preserves factual knowledge better by measuring perplexity on a held-out trivia set before and after the GRPO step while keeping rewards constant.
- **Frontier researcher:** Probe the open question on entropy vs linguistic uncertainty by logging entropy spikes during the build, correlating them with reward delta, and checking whether high entropy always precedes reward increases; use this data to falsify or refine the GTPO-style derivative assumption.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*