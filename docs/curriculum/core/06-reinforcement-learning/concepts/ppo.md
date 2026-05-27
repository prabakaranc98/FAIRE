---
title: Proximal Policy Optimization
slug: ppo
layer: core
subject: 06-reinforcement-learning
page_type: concept
state: drafted
authors_anchored: [bai, yu, kempner, ouyang]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [policy-gradients, advantage-estimation, rl-training-loop, reward-modeling]
tags: [ppo, rlhf, clipped-objective, gae, policy-gradient, critic-regularization]
updated: 2025-06-10
has_mvb: true
---

Imagine you are asked to improve a helpful assistant, but every time you want to push it toward a new reward signal—say, politeness over bluntness—it costs not only the policy weights but also another equally large critic to stay stable. The critic sits in GPU memory alongside the policy, the replay buffer grows, and suddenly your 70B-parameter alignment job demands twice the VRAM of inference alone. The only reason this is tolerable in current RLHF practice is that Proximal Policy Optimization (PPO) quietly trades that memory tax for a clipped gradient and a soft trust region. By the end of this page, you will be able to explain why the clipping term, the value critic, the KL penalty, and Generalized Advantage Estimation (GAE) nail down this stability, how recent critic-light papers rediscover the same regularization at half the compute, and what a lean Colab PPO loop feels like when you run it end-to-end.

## The territory

The RLHF stack lives squarely inside policy gradients: a policy generates language, a reward model assigns scores, and the gradient estimator pushes probability mass toward high-score responses. Early RLHF systems copied Trust Region Policy Optimization (TRPO) to guarantee monotonic improvement, but TRPO’s constrained line-search and complex Hessians do not scale to the 1× cost budget of production alignment. PPO arrived as a pragmatic compromise—rather than solving a constrained optimization, it uses the ratio between new and old policy probabilities to clip updates and keep them within a soft trust region. That simple trick kept policy gradients in check, so researchers could afford to run PPO with a value-function critic and reward model without an explicit inverse Hessian. PPO’s clipped objective, critic supervision, and optional KL penalty form a family of methods that balance exploration with the off-policy stability of actor-critic algorithms while remaining implementable with minibatch SGD.

In practice, PPO answers the question: how do you keep a policy from “overshooting” when your reward signal comes from a learned model instead of a known simulator? The clipping op prevents large ratios, the critic gives a baseline to shrink variance, and the KL penalty anchors you to a reference policy so the reward model doesn’t become a torque box. Because these ingredients each have a computational cost—especially the value critic whose architecture often matches the policy’s size—researchers are now asking whether the same regularization can be obtained with less overhead. This story about balancing stability, compute, and RLHF readiness is why PPO is still taught as the alignment workhorse even as critic-free alternatives surface. How does its mechanism look once we lay out the equations and the runtime loop?

## How it works

PPO’s core mechanism is an objective that punishes updates that change the policy too much while still following the advantage signal. Start with the policy ratio
\[
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
\]

where \(\pi_\theta\) is the current policy parameterized by \(\theta\), \(\pi_{\theta_{\text{old}}}\) is the policy that generated the data, \(a_t\) is the chosen action (which in language alignment is the next token), and \(s_t\) is the prefix context or environment state. The surrogate loss clips this ratio:
\[
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \, \hat{A}_t,
\text{clip}\!\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right) \, \hat{A}_t \right)\right]
\]

where \(\hat{A}_t\) is the estimated advantage at timestep \(t\), \(\epsilon\) is the clip threshold (typically 0.1–0.3), and \(\text{clip}\) constrains \(r_t\) within \([1-\epsilon, 1+\epsilon]\). The min operator selects the conservative estimate: if the ratio tries to push too hard in the direction of positive advantage, clipping keeps the update tied to the old policy; if the ratio goes below \(1-\epsilon\), the same clip prevents collapses. In other words, clipping acts as a soft trust region, keeping the policy update within a band where Taylor approximations to the KL divergence remain valid.

Each policy gradient step still needs \(\hat{A}_t\), and this is where the critic enters. PPO usually pairs the clipped loss with a value-function loss,
\[
\mathcal{L}^V(\theta) = \mathbb{E}_t\left[(V_\theta(s_t) - V^{\text{target}}_t)^2\right]
\]

where \(V_\theta(s_t)\) is the critic’s output for state \(s_t\), and \(V^{\text{target}}_t\) is the truncated multi-step return computed with GAE. The Generalized Advantage Estimation formula
\[
\hat{A}_t^{\text{GAE}(\lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
\]

is where each TD error \(\delta_{t} = r_t + \gamma V_{\theta_{\text{old}}}(s_{t+1}) - V_{\theta_{\text{old}}}(s_t)\); \(\gamma\) is the discount factor and \(\lambda\) controls bias-variance trade-off. GAE sums discounted TD errors so longer horizons get more weight when \(\lambda\) is near 0.95, reducing variance compared to Monte Carlo returns while still keeping the critic’s bias manageable. In text alignment, we usually set \(\gamma = 1\) and limit episodes via sequence length, making the advantage mostly a function of reward minus the critic baseline.

The critic is expensive because it must either be a copy of the policy network (when sharing the transformer body) or a standalone head that sees every prefix. Each RLHF step now requires forward passes through both the policy (to compute log-probabilities for the ratio) and the critic (to compute the value target). Worse, the critic may itself be finetuned on the same 70B model architecture, doubling the weight storage in GPU memory and forcing mixed-precision engineering to avoid OOMs. That is the “critic tax” introduced in the hook: the stability payoff is real, but so is the hardware demand.

To push PPO toward more practical settings, practitioners add one more regularizer—a KL penalty that anchors the updated policy to a reference policy. The full PPO loss becomes
\[
\mathcal{L}^{\text{PPO}}(\theta) = -\mathcal{L}^{\text{CLIP}}(\theta) + c_1 \mathcal{L}^V(\theta) + c_2 \mathbb{E}_t\left[\text{KL}\left(\pi_{\theta_{\text{old}}}(\cdot \mid s_t) \, \| \, \pi_\theta(\cdot \mid s_t)\right)\right]
\]

where \(c_1\) and \(c_2\) balance value regression and KL penalty, respectively. The KL term is often implemented using a running hinge: if the averaged KL exceeds a threshold (for example 0.01), the gradient update is scaled down to stay below the threshold. This is how Bai et al. (2022) [arxiv:2204.05862] trained helpful and harmless assistants—by letting one reward model express “useful” behavior and the other express “harmless” behavior while PPO’s KL kept the policy from drifting into any single reward lead. Their paper is foundational for scaling PPO to RLHF because it showed how to combine those competing rewards while still updating on partial sequences with clipping, GAE, and the KL term.

Since the KL penalty reuses the reference policy’s logits, engineers can adjust it dynamically per batch, pausing gradient updates when the policy enters an unexplored region far from \(\pi_{\theta_{\text{old}}}\). The policy ratio \(r_t\) and the KL penalty both come from the same probability tensor, so the implementation is straightforward: after sampling a batch of rollouts, compute log-probabilities from the current policy, subtract log-probabilities from the reference policy, and derive advantages from the critic. GAE reduces variance, clipping prevents large steps, and the value loss regularizes with a baseline—this trio is the mechanical core.

To sample data for PPO, you need a policy, an environment, and a reward model. In RLHF on language, the environment is usually the token-generation procedure itself: you feed a prefix into the policy, let it sample the next \(k\) tokens, and treat the resulting sequence as the “action.” The reward model evaluates the entire sequence or the final token. The critic sees every state \(s_t\) (a prefix) and outputs values that the policy uses as baselines. When you implement this loop, you alternate between collecting new rollouts with the current policy (without gradient updates) and running PPO updates on the stored batch; because the stored data is on-policy, the “old” policy is the one used when the data was collected, and you recompute the ratio \(r_t\) with that snapshot.

Modern work pushes this loop in two directions. On the research side, the 2025 paper by Yu et al. [arxiv:2505.18531] studies KL-regularized policy architectures and shows how different entropy schedules interact with PPO’s clipping, particularly on reasoning benchmarks. They show that a curriculum of gradually decreasing the KL coefficient prevents entropy collapse while still allowing the policy to move quickly during early training. Their formulation keeps the same surrogate objectives but applies KL regularization on grouped tokens (for example, reasoning steps), demonstrating that PPO’s stability arises from the local trust region implied by the clipping, not necessarily from the critic itself. This insight paves the way for lightweight alternatives.

The Kempner Institute’s recent work “Untitled” [arxiv:2405.07863] introduces A*-PO, which reimagines PPO without an online critic. Instead of learning \(V_\theta(s_t)\), A*-PO directly regresses \(\hat{A}_t\) toward a precomputed “optimal” advantage derived from offline search. Because the regression target is fixed, you only need the policy network; there is no critic to update or store. The same clipped surrogate objective emerges because the advantages still figure into \(\mathcal{L}^{\text{CLIP}}\). A*-PO demonstrates that clipping already enforces regularization; the critic is a tool to estimate the advantage but is not structurally required for PPO-level stability. The compute footprint hence drops dramatically, but you must provide those offline advantage targets, which is currently practical only in constrained environments.

## Where the field is now

Today’s alignment labs still lean on PPO as their baseline, but the plumbing keeps changing. Researchwise, Online Iterative Reinforcement Learning from Human Feedback with General Preferences (2024) [arxiv:2402.07314] replaced the static reward model in a PPO loop with a streaming human preference dataset. Instead of collecting rewards once, they alternate between policy updates and online preference aggregation, dynamically adjusting the reward network so it never strays far from the latest human judgments. PPO’s clipping ensures the policy can update safely even while the reward signal keeps shifting under its feet, which is why they still report stable convergence on preference-ranked datasets. The same paper spins off a “general preference” reward model that can score any pair of responses, and PPO handles that global ranking by keeping \(r_t\) bounded over the entire pipeline.

Engineering teams are also investing in efficient PPO infrastructure. OpenAI’s research blog “Training language models to follow instructions” (openai.com/research/learning-from-human-feedback) documents a multi-stage PPO stack: they collect response rankings, train a reward model, precompute value targets with a multi-step critic, and run distributed PPO on TPUs with gradient accumulation tuned to 0.005 KL thresholds. The engineering frontier is now about scaling the critic and policy in lockstep while keeping the KL penalty as an “emergency brake”—the same formula as in Bai et al. doesn’t change, but the system now includes asynchronous data pipelines, gradient-checkpointed models, and streaming reward updates. The production claim is clear: RLHF still uses PPO, but it pairs the algorithm with infrastructure that squeezes memory usage and reduces variance with massive batch sizes.

On the research frontier, Yu et al. (2025) [arxiv:2505.18531] and Kempner’s A*-PO [arxiv:2405.07863] have already shown that the stability of PPO is captured by the clipped surrogate; the critic is mainly there to supply advantages. As practitioners, we now ask whether we can safely replace expensive critic updates with lighter estimators or even offline advantage targets while keeping the clipping and KL terms intact. At the same time, RLHF practitioners continue to deploy PPO in production because the algorithm is the only one that both stabilizes gradients and scales through token-level logits at 70B parameters without the Hessian costs of TRPO.

## What's still open

Can we derive token-level KL penalties that mimic PPO’s clipping while operating on partial sequences without requiring a separate critic? The policy ratio and KL penalty both look at entire tokens, but in long-form reasoning the reward for token \(t\) should depend on outcomes tens of tokens later, and the critic cannot feasibly render such future-aware baselines without exploring the entire rollout again.

Does the advantage regression in A*-PO generalize to open-ended language if we cannot compute “optimal” offline advantages? If a reward model is learned online, any fixed offline target is stale; the open question is whether we can recompute approximate advantages with a lightweight surrogate that still respects PPO’s trust region.

How can we instrument PPO’s TensorFlow-style rollout data pipeline so that per-token gradients track human preferences without overwhelming GPU memory? Every new prefix doubles the critic’s tape, so the challenge is building streaming accumulation of \(\delta_t\) and \(\hat{A}_t\) without dumping the entire rollout batch in one shot.

Is there an architecture that shares weights between policy and critic at the execution level while allowing each to have different precision (e.g., 4-bit policy, 8-bit critic) so the memory footprint of the critic tax is halved without losing advantage fidelity? Solving this would bridge today’s PPO implementations and the critic-free variants that claim half the compute.

## Where to read next

If you want the probabilistic foundation, → [[policy-gradients]] traces the derivation of the policy gradient theorem that PPO clips, and if you prefer the variance reduction story, → [[advantage-estimation]] describes TD(λ) and GAE in detail. For the engineering counterpart that keeps the KL trust region low-latency, → [[rlhf-infrastructure]] shows how data pipelines and reward models form the backend that PPO protects.

## Build it

Building the PPO loop yourself is the single best way to feel the critic tax. This recipe keeps everything in one script, executes on a single Colab T4, and surfaces the interplay of clipping, GAE, and KL regularization while letting you see how a critic eats memory.

**What you're building:** a bare-metal RLHF loop that finetunes Qwen-1.5-0.5B on an IMDb-style sentiment preference task using PPO with clipping, GAE, and a KL penalty.

**Why this is valuable:** you will manually compute \(r_t\), apply the clip, regress a critic for \(\hat{A}_t\), and optionally swap to the critic-free advantage regression propped up by Paddle’s OCR detectors so you understand the “what broke” part of critic tax.

**Stack:**
- **Model:** HuggingFace Qwen/Qwen-1.5-0.5B (policy) plus PaddlePaddle/PP-OCRv5_server_det and PaddlePaddle/PP-OCRv5_server_rec for reward instrumentation that mimics pairwise preferences with synthetic OCR scores.
- **Dataset:** `imdb` (HuggingFace) for sentiment labels and prompts.
- **Framework:** PyTorch 2.2 + Accelerate 0.26 + Transformers 4.41 + bitsandbytes 0.41.
- **Compute:** Google Colab T4 (16 GB VRAM), expect ~2.5 hours for 3k gradient steps with gradient accumulation.

**The recipe:**
1. Install the stack with `pip install torch torchvision accelerate transformers bitsandbytes datasets evaluate wandb`. Initialize Accelerate CLI for your T4 and set gradient accumulation to 4 so token context fits 16 GB.
2. Load IMDb prompts, tokenize with Qwen’s tokenizer, and build episodes of 256 tokens. Compute a synthetic reward by generating two candidate completions (baseline vs. policy) and score them through the OCR detection/recognition pipeline: run PaddlePaddle/PP-OCRv5_server_det to get bounding boxes, then PaddlePaddle/PP-OCRv5_server_rec to transcribe; higher detection confidence plus lower OCR error means higher reward.
3. Implement GAE: compute TD errors with \(\delta_t = r_t + \gamma V_\theta(s_{t+1}) - V_\theta(s_t)\), set \(\gamma=1\), \(\lambda=0.95\), and accumulate advantages over the rollout. Normalize advantages across the batch before using them in the surrogate loss.
4. Train the PPO loop for 3k steps with clip range \(\epsilon=0.2\), critic loss weight \(c_1=0.5\), KL penalty weight \(c_2=0.01\), learning rate \(2.5\times 10^{-5}\), and minibatch size 256. Monitor the clipped surrogate loss and the mean KL; if the KL exceeds 0.015, scale the LR by 0.5 for the next 100 updates.
5. Evaluate by generating replies to a held-out sentiment prompt set and checking that PPO’s reward evaluator now prefers policy completions over the reference baseline at least 70% of the time.

**Expected outcome:** a single-file checkpoint of aligned Qwen-1.5-0.5B, paired with a critic checkpoint and a rewards/metrics log showing clipped loss, critic loss, and KL per step.

- **CS student:** Swap Qwen for a 1.5B distilled version, skip Paddle’s OCR models (use a lightweight sentiment classifier as reward), and shorten training to 1.5k steps—this keeps the experience within a Colab runtime.
- **Applied engineer:** Quantize the policy to 4-bit using bitsandbytes and serve it with a vLLM HTTP endpoint; gather latency at p50 < 350 ms while running the actor-critic update loop asynchronously so inference and gradient steps share the same GPU.
- **Applied researcher:** Ablate \(c_2\) (KL weight) by training two copies—one with \(c_2=0.01\), another with \(c_2=0.05\)—and report how the token-level entropy and reward preference accuracy diverge, testing the hypothesis that a stronger KL keeps the policy closer to the reward model.
- **Frontier researcher:** Replace the critic regression target with a synthetic “optimal” advantage produced by replaying the PPO rollouts through the reward evaluator and tune whether A*-PO-style regression keeps the surrogate loss stable; the falsifier: if the critic-free variant’s KL exceeds 0.025 before 2k steps, the approach has not matched PPO’s stability.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*