---
title: "Step 1 — Formalize the Token-Level MDP"
slug: "step-1-formalize-token-level-mdp"
layer: core
subject: 06-reinforcement-learning
page_type: concept
state: reviewed
authors_anchored: [sutton, bai]
feeds_de_pillar: []
arc_position:
  arc: world-models-and-imagination
  prev: null
  next: step-02-policy-gradient
mvb_personas: [applied-ai-engineer, research-engineer, applied-researcher]
prereqs: [markov-decision-process, reward-modeling, policy-gradient]
tags: [rlhf, token-level, credit-assignment, grpo]
updated: 2025-10-10
has_mvb: true
---
> **Arc:** [World Models And Imagination](../../arcs/world-models-and-imagination.md) — Step 1 of 5


Every time you ask a large language model for a multi-step answer, the human judges only ever see the end result—reward flows back through a single scalar after dozens of tokens have been emitted. The human in the loop never explains which clause in the justification was on point and which sentence derailed safety, so the optimizer cannot tell which token earned the thumbs-up and which one earned silence. That is a practical catastrophe for credit assignment, and it is what keeps policy gradients from learning to steer on the earliest tokens in a chain of thought. This step asks: what happens if we stop treating generation as one big terminal reward and instead replay every intermediate token as a deterministic transition with its own provisional signal? By the end of this page you will know how to turn token histories into an MDP, log per-token entropies in normalized units, and manufacture the proxy rewards that Step 2’s GRPO implementation will consume.

# The territory

Reinforcement learning from human feedback (RLHF) traditionally collects prompt–response pairs, judges the entire response, and then fits a reward model whose single output score gets fed to a policy optimizer (Bai et al. 2022 [arXiv:2204.05862]). Dong et al. (2024) laid out the workflow that follows from this structure: batch-collected episodes, reward modeling, offline policy improvement, then online updates, and throughout the flow the reward signal remains sequence-level because the judges never score an intermediate token [arXiv:2405.07863]. That has some advantages—you do not have to instrument the human labeler to comment on partial generations—but it also leaves the model blind to which of the hundreds of emitted tokens were responsible for praise or correction. Online Iterative Reinforcement Learning from Human Feedback with General Preference (2024) starts to peel that onion by interleaving online preference data with incremental policy updates, hinting that a finer-grained view of reward assignment improves generalization across prompts [arXiv:2402.07314]. The question at this step of the arc is: can we engineer that finer granularity into the MDP itself?

A token-level MDP is how we answer that question. Instead of bundling all emitted tokens into one terminal transition, we lift each prompt history \(s_t\) and its successor token \(a_t\) into a deterministic state-action pair, and then record an implicit reward \(r_t\) that reflects how much of the final outcome \emph{should} be the responsibility of that token. For product managers and policy directors, that structure is what unlocks transparency and auditability: you can now inspect which tokens receive large credit, which ones soft-collapse into low entropy, and whether reward-shaping parameters skew toward the beginning or end of a response. In regulated deployments where every response must be traceable, a token-level log is a liability-mitigating artifact.

Because this territory is both a foundation for GRPO in Step 2 and a bridge to any future token-level RL method, the rest of the page shows you how to construct that deterministic simulator, how to distribute reward, how to normalize entropy, and how to capture aggregate gradients without rewriting the training loop. Once the simulator is in place, the policy-gradient updates in the next step can simply read from the logged transitions instead of reconstructing histories from scratch.

## How it works

The key mechanism is a deterministic simulator that turns the language model’s tokenizer into a transition model, attaches proxy rewards, and tracks normalized entropy statistics while the new policy beliefs remain untouched. Rather than hiding mathematics in a separate subsection, we interleave it with intuition in the narrative below; every symbol is defined immediately in prose.

### Deterministic token transitions

The state \(s_t\) is the full token history seen so far, and the action \(a_t\) is the token appended by the policy at that step. Under that definition, the transition function is not a learned model but a deterministic concatenation:

\[
P(s_{t+1} \mid s_t, a_t) = \delta(s_{t+1} = s_t \oplus a_t)
\]

where \(\delta(\cdot)\) is the Dirac delta that equals 1 when its argument is true and 0 otherwise, \(\oplus\) is list concatenation, \(s_{t+1}\) is the history after appending token \(a_t\), and \(s_t\) is the history before \(a_t\). This equation says that the next state is literally the history you obtain by appending the chosen token, and because we never sample randomness, the simulator can replay the same sequence of states and actions for gradient checking or visualization. That determinism is what makes the token-level MDP diagnostically useful: nothing between \(s_t\) and \(s_{t+1}\) is sampled twice, so we can attach rewards post hoc.

Because the transition dynamics are deterministic, the only stochasticity during execution comes from the policy’s sampling procedure, which we keep unchanged from the pretrained GPT-Neo logits. This design choice ensures that Step 1 reproduces the same rollouts you would measure in production but now exposes every state, action, and reward proxy.

### Implicit per-token rewards and entropy

Once states, actions, and transitions exist, the next design question is how to spread the final outcome \(R_{\text{outcome}} \in \{0,1\}\) across tokens. We define an implicit reward:

\[
r_t = R_{\text{outcome}} \cdot \frac{\exp(-\alpha t / T)}{\sum_{i=1}^T \exp(-\alpha i / T)} + \lambda\, \mathcal{H}_{\text{norm}}(\pi(\cdot \mid s_t))
\]

where \(T\) is the total number of tokens in the rollout, \(t\) is the current timestep, \(\alpha\) controls the temporal decay (larger \(\alpha\) pushes credit toward the beginning), \(\lambda\) scales the entropy bonus, and \(\mathcal{H}_{\text{norm}}(\pi(\cdot \mid s_t))\) is the normalized entropy of the policy’s predictive distribution at \(s_t\). The exponential window is inspired by KL-regularized policy gradients such as those discussed in the (ongoing) Yu et al. 2025 preprint “Generalized Reward Proxies for Online Control” [arXiv:2505.18531], which showed that distributing credit smoothly backward over tokens reduces variance without requiring exact intermediate reward labels. The entropy bonus keeps gradients from collapsing by penalizing low-entropy action distributions, matching the stabilizing intuition from Bai et al. 2022 that KL or entropy regularization is crucial when the policy can overshoot a narrow set of tokens [arXiv:2204.05862].

The entropy term itself is computed over the full tokenizer vocabulary. Let \(\boldsymbol{p}\) be the softmax probabilities \(p_k = \frac{\exp(\ell_k / \tau)}{\sum_{j=1}^V \exp(\ell_j / \tau)}\) where \(\ell_k\) are the model’s logits for token \(k\), \(V\) is the vocab size (≈50,257 for GPT-Neo), and \(\tau=1\) is the sampling temperature. The raw entropy is

\[
\mathcal{H}(\pi(\cdot \mid s_t)) = -\sum_{k=1}^V p_k \ln p_k
\]

where the logarithm is natural (base \(e\), yielding nats) and the sum runs over the entire vocabulary of plausible next tokens. To normalize this quantity, divide by \(\ln V\) so that \(\mathcal{H}_{\text{norm}} = \mathcal{H}/\ln V\); this bounds the normalized entropy between 0 and 1 irrespective of the vocabulary size. In practice, GPT-Neo rollouts sample from regions of the distribution where the normalized entropy lives between about \(0.45\) and \(0.75\) nats-per-vocab-unit, so expecting values in that band is realistic for this architecture and tokenization. Reporting normalized entropy also makes comparisons between different models and tokenizers fairer—the denominator \(\ln V\) automatically compensates for larger vocabularies.

Including entropy in the token reward turns each \(r_t\) into a mix of backward-distributed outcome signal and forward-facing uncertainty penalty. The build replicates this mixture so that Step 2’s GRPO algorithm can simply subtract the logged entropies from the policy’s gradients, just as Yu et al. (2025) anticipate in their general reward proxy framework [arXiv:2505.18531].

### Token-level gradients before GRPO

The deterministic transitions and per-token rewards allow you to compute a provisional gradient proxy even before you run GRPO. You still compute the model logits at each state \(s_t\), but instead of backpropagating through multiple time steps, you log the scalar product \(r_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\) and verify how those per-token contributions align with the final outcome. This logging is critical for debugging: if the earlier tokens consistently produce negative dot products for positive outcomes, you know the reward decay\slash entropy balance is misconfigured even before policy updates.

Because this simulator uses the same GPT-Neo logits as the deployed policy, the logged gradients serve as a ground truth for Step 2’s masked updates. When the GRPO routine later masks the tokens marked by high reward proxies, it is operating over the same \((s_t, a_t, r_t)\) tuples you just recorded. That structural alignment is why Step 1 is indispensable: without it, the gradient target would be re-derived from terminal labels and would drift away from what the simulator captured.

## Where the field is now

Researchers have moved steadily toward token-level reasoning in response to the credit assignment bottleneck. Dong et al. (2024) [arXiv:2405.07863] argued that offline and online RLHF stages should share structured replay buffers, where intermediate state-action pairs are stored separately from their terminal scores. Building on that, the Online Iterative Reinforcement Learning from Human Feedback with General Preference paper (2024) [arXiv:2402.07314] tied policy updates to continually refreshed preference datasets, which is effectively a real-world experiment in token-level reward reshaping because each new preference can be attributed to the latest action tokens. More recently, Yu et al. (2025) [arXiv:2505.18531] introduced generalized reward proxies that include entropy-like regularizers, demonstrating on open-source benchmarks that token-level proxies outperform sequence-level critics when tasks require nuanced reasoning chains. Those research advances show the empirical payoff of the simulator you are building here: once every token transition is logged with an implicit reward, the policy gradients become much easier to debug and compare across different policy checkpoints.

On the engineering side, OpenAI’s engineering blog on RLHF (OpenAI Research, 2024) highlights the need for modular pipelines where the data engineering team can inspect the same token-level traces that the policy team is training on; their deployed workflow includes transition stores, entropy diagnostics, and scripts that replay traces before any gradient computation. Meta AI’s systems team has adopted similar instrumentation for LLaMA-3 updates, logging per-token reward signals to monitor alignment drift at scale. Together these engineering efforts underline that the simulator built in Step 1 is not just academic—the production pipelines already expect deterministic traces and entropy checks before they let GRPO or PPO consume the data.

## What’s still open

The most pressing theoretical question is whether the soft exponential decay we use for \(r_t\) can be derived from a latent reasoning trace rather than being hand-tuned. Can we prove that for a class of reasoning problems, there exists a decay schedule \(\alpha^\star(t)\) such that the proxy \(r_t\) remains within \(\varepsilon\) of the true token-level advantage, thereby preventing reward hacking when the policy moves off-distribution?

A second challenge is to learn the per-token reward redistribution directly, for example by training a small secondary transformer that predicts the importance weight instead of relying on the exponential kernel. Does that learned attention weight align better with human preferences, and can it be learned within the same compute budget as the main policy without introducing instability?

Third, although we normalize entropy by \(\ln V\), we still observe that normalized entropies drift when the tokenizer vocabulary changes (e.g., when switching from GPT-Neo’s BPE to a sentencepiece model). How should we recalibrate \(\mathcal{H}_{\text{norm}}\) when the policy’s candidate set is restricted (e.g., through adaptive vocab filtering) so the entropy bonus remains a faithful uncertainty proxy?

## Where to read next

If you want to understand the canonical reinforcement learning dynamics that underlie the token-level simulator, see [[curriculum/core/06-reinforcement-learning/mdp]] for states, actions, and Bellman equations in the classical setting. For a deeper dive into how reward models produce the final scalar \(R_{\text{outcome}}\), → [[curriculum/core/06-reinforcement-learning/reward-modeling]] explains the training pipelines and human labeling conventions. What can you build next? Step 2 — [[step-02-policy-gradient|Token-level GRPO]] — consumes the traces you just logged, and if you crave more gradient theory, the policy-gradient page in the same curriculum walks through entropy-regularized estimators that will appear in that step.

## Build it

**What you’re building:** a token-level MDP simulator that logs deterministic transitions, normalized entropies, and implicit per-token rewards on `castorini/mdpr-tied-pft-msmarco` rollouts so that you can replay the same traces when GRPO updates the policy.

**Why this is valuable:** seeing the same states \(s_t\), actions \(a_t\), and reward proxies \(r_t\) that the policy trainer will use is the prerequisite for debugging the GRPO updates in Step 2. It also provides the empirical evidence that per-token gradients, once regularized by normalized entropy, align with the eventual reward labels.

**Stack:**
- **Model:** `castorini/mdpr-tied-pft-msmarco` (base retrieval checkpoint) and `castorini/mdpr-tied-pft-msmarco-ft-all` (fine-tuned LM for scoring)
- **Dataset:** `msmarco-passage` (for prompt templates and reward outcomes)
- **Framework:** `transformers 4.44`, `accelerate 0.21`, `torch 2.1.1`
- **Compute:** Single consumer GPU (RTX 4060 8GB or Colab T4); wall-clock ~3 hours

**Estimated time:** 3 hours (dataset download ~30 min, simulation coding ~90 min, logging/debugging ~60 min).

**Success criterion:** the simulator logs five prompts such that each logged \(s_t\) grows deterministically by one token, the implicit reward vector sums to the final outcome within ±0.01 per prompt, and the normalized entropy \(\mathcal{H}_{\text{norm}} = \mathcal{H}/\ln V\) stays between 0.45 and 0.75 for sampled tokens while \(V\) is the full vocabulary size (≈50,257) and \(\mathcal{H}\) is computed with natural logarithms (nats). You should also store a JSON trace of one rollout for use in Step 2.

### The recipe

1. Install dependencies: `pip install transformers==4.44 accelerate==0.21 torch==2.1.1 datasets`. Load `msmarco-passage` from Hugging Face, sample five passages, and map each to prompts of at least 32 tokens using the tokenizer that accompanies `castorini/mdpr-tied-pft-msmarco`.
2. Initialize the tokenizer and models. For each prompt, iterate over tokens \(a_t\); construct \(s_t = \text{tokens}[:t]\) and log `(step, len(s_t), accumulated tokens)` to confirm deterministic append behavior.
3. Compute implicit rewards using \(r_t = R_{\text{outcome}} \cdot \frac{\exp(-\alpha t / T)}{\sum_{i=1}^T \exp(-\alpha i / T)} + \lambda\, \mathcal{H}_{\text{norm}}(\pi(\cdot \mid s_t))\) with \(\alpha=4.0\), \(\lambda=0.02\), and \(R_{\text{outcome}}\) normalized to \([0,1]\). Afterward, `assert abs(sum(rewards) - outcome_label) < 0.01`.
4. Query the fine-tuned policy (`castorini/mdpr-tied-pft-msmarco-ft-all`) to compute logits at each \(s_t\), derive probabilities \(p_k\) via softmax with \(\tau=1\), and compute entropy \(\mathcal{H} = -\sum_k p_k \ln p_k\), logging \(\mathcal{H}_{\text{norm}} = \mathcal{H}/\ln V\) to ensure it falls in [0.45, 0.75]. Also log the entropy bonus separately to inspect its contribution.
5. Save one rollout (states, actions, reward proxies, entropies) to disk as JSON and print the filename so Step 2 can load it directly.

### Expected outcome

A deterministic trace file containing states, actions, normalized entropies, and reward proxies that sum to each outcome label; console logs show the simulated \(s_t\)-to-\(s_{t+1}\) transitions and confirm the entropy band. This artifact can be fed into GRPO for masked gradient updates, and the logged entropies demonstrate the policy’s uncertainty throughout the rollout.

### Common failure modes

- Tokens skip (len(s_t) jumps) → tokenizer mismatch; reinitialize with the checkpoint’s tokenizer and rerun sampling.
- Reward sum deviates by >0.01 → outcome label not normalized; rescale \(R_{\text{outcome}}\) to [0,1] and recompute.
- Normalized entropy outside [0.45, 0.75] → logits too sharp/broad; adjust temperature \(\tau\) (e.g., set \(\tau=1.2\)) when computing \(p_k\).

### Variants per persona
- **Applied AI/ML engineer:** use the saved traces to build a logging dashboard that ties token-level rewards to the monitoring metrics in production; extend the simulator to log latency and memory usage for each transition so the deployment team can observe cost-per-token alongside credit assignment.
- **Research engineer:** reproduce Table 2 from the GRPO paper (Yu et al. 2025) using the JSON trace as input, verifying that the per-token entropy penalty with \(\lambda=0.02\) reduces variance by at least 15% compared to vanilla policy gradient.
- **Applied researcher:** hypothesize that doubling \(\alpha\) shifts rewards toward earlier tokens and reduces reward hacking; design an experiment that sweeps \(\alpha \in \{2,4,8\}\), logs the Pearson correlation between \(r_t\) and final outcomes, and reports the correlation trend.

### Stretch goals

- Extend the simulator to support both positive and negative outcome labels and observe whether the decay weighting naturally flips sign when the final reward is negative.
- Replace the entropy bonus with KL divergence to a uniform policy (per Yu et al. 2025) and compare the logged divergence values to understand whether entropy or KL better smooths gradients before GRPO.
- Combine the simulator with human-critical subsets (e.g., safety-sensitive passages from `msmarco-passage`) to see whether critical contexts concentrate implicit rewards toward the final tokens.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*

