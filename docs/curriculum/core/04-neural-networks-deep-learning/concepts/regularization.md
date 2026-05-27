---
title: Regularization in Large Model Fine-Tuning
slug: regularization-large-models
layer: core
subject: 04-neural-networks-deep-learning
page_type: concept
state: drafted
authors_anchored: [bengio, lecun, goodfellow, radford]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [policy-gradient, rlhf, kl-regularization, trajectory-optimization]
tags: [regularization, reinforcement-learning, policy-gradient, activation-sketching, kl-divergence, freezing]
updated: 2025-12-09
has_mvb: true
---

# Regularization in Large Model Fine-Tuning

Imagine a world-class gymnast being asked to learn a new dance routine by rewiring their muscles mid-flight; the new routine might land, but their ability to walk, jump, and twist collapses. That is what happens when you run GRPO or supervised fine-tuning (SFT) on a frozen base model without constraints: the policy excels on the fresh reward signal, yet its factual knowledge, reasoning stability, and few-shot generalization fray overnight. A March 2026 preprint demonstrated this surgical ruin, showing that raw fine-tuning on a new reasoning dataset drives the output distribution to collapse into a handful of confident, but factually empty, modes (Author et al. 2026) [https://www.arxiv.org/pdf/2603.18168]. The regularizers that once meant weight decay now need to act like tendons, anchoring the trajectory of the entire distribution while still letting novel capabilities bloom. By the end of this page readers will understand how KL-penalty anchors, selective freezing, and activation sketching make that possible, which ones matter in real-world RLHF, and how to code a lightweight KL-regularized GRPO step that visibly charts the drift between reference and fine-tuned policies.

## The territory

Large language models are pretrained on massive corpora, and their emergent capabilities become the baseline that downstream tasks build on. The central problem regularization answers is: how can we adapt these models to new tasks without throwing away the base distribution? In classical settings the answer was a static penalty term—L2 weight decay, dropout, early stopping—to keep parameters from wandering too far. Modern large-model fine-tuning reframes the question as a trajectory constraint. Instead of punishing parameter magnitude alone, it compares the entire output distribution of the fine-tuned policy \( \pi_\theta \) to a stabilized reference \( \pi_{\mathrm{ref}} \) (often the pretrained or preceding policy). The regularizer becomes a dynamic anchor: a KL divergence term, a set of frozen layers, or even an activation sketch that keeps the bases of knowledge intact while gradients chase new rewards. This family of methods borrows language from information theory (KL for divergences) and from systems engineering (activation sketching for bandwidth-aware constraints). Before walking through the specific mechanisms, note that the engineering constraint is to make these anchors cheap enough to run on accessible hardware like a Colab T4, while the research constraint is to keep them tight enough to prevent forgetting but flexible enough to let exploration happen. How does this trajectory anchoring actually work?

## How it works

### Anchoring the trajectory with KL penalties

A policy gradient update without anchors only maximizes expected advantage, which encourages \(\pi_\theta\) to dart toward greedy actions on the new reward. The KL-regularized objective introduces a penalty term that measures how far the new policy diverges from the reference. The key idea is to rewrite the loss as

\[
L(\theta) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\theta(\cdot \mid s)}\left[-\hat{A}(s,a) + \beta D_{\mathrm{KL}}(\pi_\theta(\cdot \mid s) \,\|\, \pi_{\mathrm{ref}}(\cdot \mid s))\right]
\]

where \(s\) is a prompt-state sampled from the fine-tuning dataset \(\mathcal{D}\), \(a\) is the sampled action (token sequence) from the current policy \(\pi_\theta\), \(\hat{A}(s,a)\) is the advantage estimate, \(\beta > 0\) is the KL coefficient, and \(D_{\mathrm{KL}}\) is the forward KL divergence. This term pushes \(\pi_\theta\) toward \(\pi_{\mathrm{ref}}\) when \(\beta\) is large, preventing the policy from assigning high probability mass to actions the reference would never take.

The distinction between forward and reverse KL matters: On the Design of KL-Regularized Policy Gradient Algorithms (Author et al. 2025) [https://arxiv.org/abs/2505.17508] shows that using \(D_{\mathrm{KL}}(\pi_{\mathrm{ref}} \,\|\, \pi_\theta)\) (reverse KL) biases the update toward covering reference support, which is useful when the reference already captures the desired behavior, whereas forward KL prioritizes staying close to high-probability reference actions but allows broader exploration. In practice, implementations mix both: the RLHF step that penalizes both \(\pi_\theta\) w.r.t. \(\pi_{\mathrm{ref}}\) and vice versa yields more stable reasoning while still experimenting with novel generations. The paper also formalizes how the KL coefficient \(\beta\) acts like a temperature on the policy interpolation, and how adjusting \(\beta\) over the gradient step can balance exploration with constraint compliance. The upcoming MVB section will let you sweep \(\beta\) yourself and plot the resulting drift.

### Parameter freezing and selective scaling

KL constraints keep distributions similar, but some parameters are disproportionately responsible for knowledge storage. The Scalpel vs. Hammer study (Author et al. 2025) [https://arxiv.org/abs/2507.10616] argues that indiscriminate freezing (the hammer) loses too much adaptability, while fine-grained scaling (the scalpel) of specific layers or attention heads preserves factual recall while still letting deeper layers adjust. The trick is to identify the subspaces that encode general knowledge—often the earlier layers—and clamp their updates either to zero (hard freezing) or to dampened gradients (scaling). The paper introduces an adaptive scalar \( \alpha_l \in [0,1] \) per layer \(l\) such that the effective gradient is \( \alpha_l \cdot \nabla_\theta L\). Layers closer to the embedding or positional outputs get \(\alpha \approx 0\), meaning they retain their pretrained identity, while the higher layers learn new reasoning strategies. This selective freezing acts as a structural prior on the trajectory: the base capabilities remain in their original basin while only a subset of parameters stroll toward the new minima dictated by the RL signal.

Combining parameter freezing with KL penalties compounds their effect. The KL term ensures distributional similarity, while freezing ensures that the parameters that command that distribution do not diverge. In practice, engineers decay \(\beta\) over the course of training while simultaneously relaxing layer scalars, so the policy gradually gains permission to deviate when enough evidence accrues that it is exploring productive reasoning pathways.

### Activation sketching with BASIS

KL penalties and freezing operate in parameter space; activation sketching operates in representation space to regularize internal activations. BASIS: Balanced Activation Sketching with Invariant Scalars (Author et al. 2026) [https://arxiv.org/abs/2604.16324] introduces a sketching matrix \(S\) that projects activation tensors onto a smaller space, computes moments that remain invariant to scaling, and feeds these moments into a lightweight regularizer. The activation sketch is defined as

\[
u = S \cdot \phi(\mathbf{h})
\]

where \(\mathbf{h}\) is the activation tensor at a certain layer and \(\phi\) is a nonlinearity. The invariant scalar \(v\) is computed as \(v = \| u \|_2\) normalized by the sketch survival probability. The penalty term then enforces \(v\) to remain close to its reference value \(v_{\mathrm{ref}}\) via

\[
L_{\mathrm{act}} = \gamma (v - v_{\mathrm{ref}})^2,
\]

where \(\gamma\) is a small coefficient. Because the sketch \(S\) is sparse and reused across batches, the cost of this regularization is comparable to a single matrix multiplication, making it feasible in production. The invariant scalars capture the bulk geometry of activations, so the regularizer indirectly constrains the model’s intermediate states while avoiding the OOMs of storing full activations for all layers.

These activation constraints dovetail with KL penalties: while KL keeps the distribution over outputs similar, the sketch ensures that the hidden states follow a similar trajectory. In combinations, these methods lock down both ends of the transformation, making sure the forward pass stays on the pretrained manifold even while gradients at the end produce new reasoning refinements.

### Policing catastrophic drift

Catastrophic drift occurs when the fine-tuned policy's distribution becomes confident about new tokens that were never probable under \(\pi_{\mathrm{ref}}\), triggering factual collapse. Untitled (Author et al. 2026) [https://www.arxiv.org/pdf/2602.07145] quantifies this by tracking entropy drop and factuality scores as a function of unconstrained KL penalty. The study shows that even moderate RLHF steps with \(\beta = 0\) produced a precipitous fall in factuality, whereas adding a KL term with \(\beta \approx 0.1\) kept perplexity and accuracy stable. The activation sketch provides an additional signal: by monitoring \(v\) before and after the step, you can detect when hidden representations diverge into uncharted territory before outputs collapse.

Some operational pipelines layer in an adaptive controller that adjusts \(\beta\) on a per-batch basis. The controller monitors the ratio of KL to advantage and scales \(\beta\) so that the penalty never falls below a threshold relative to the reward. This kind of dynamic scaling, discussed in the next open-question section, is analogous to real-time surgical feedback: the model is performing a new skill, but the controller ensures the original muscles are never cut.

## Where the field is now

The engineering narrative today is that dynamic regularization wins the deployment battle. OpenAI's RLHF blog post (OpenAI 2024) [https://openai.com/research/rlhf] documents their KL-controller that limits divergence from the pretrained model during instruction tuning, and their scale demonstrates the infrastructure demands: the controller monitors KL on millions of prompts per day and throttles the gradient accumulation steps when the divergence grows too large. The research frontier advances this narrative with new paradigms for representing and constraining trajectories.

DDCL-INCRT: A Self-Organising Transformer with Hierarchical Prototype Structure (Author et al. 2026) [https://arxiv.org/abs/2604.01880v1] emerges as a research frontier by embedding hierarchical prototypes into the transformer body, effectively regularizing representations through prototype consistency. The prototype structure acts like a trajectory anchor at the representation level, enforcing that hidden states remain close to a small set of learned prototypes while still allowing novel compositions. This mechanism offers a complementary view to KL penalties: the model is regularized by constraining representation clusters rather than output distributions.

On the systems side, the BASIS activation sketching technique accelerates inference and regularization by compressing activations into invariant scalars (Author et al. 2026) [https://arxiv.org/abs/2604.16324]. Teams deploying instruction-tuned assistants at scale have started integrating BASIS because it sidesteps the need to store all activations for comparison, which was a bottleneck in earlier RLHF loops. The sketching approach has become a production frontier for labs that want to enforce activation-level safety without paying the memory tax of full-matrix constraints.

These developments coalesce around the insight that regularization must now be adaptive, cheap, and multi-granular—operating simultaneously on logits, parameters, and activations—to maintain base capabilities while enabling reasoning improvements.

## What's still open

1. Can we formalize a dynamic scaling schedule for the KL coefficient \(\beta\) that depends on the current KL-advantage ratio and not just wall-clock steps, so that the penalty tightens when the policy diverges and relaxes when it aligns with the reference? A controller like this would prevent reward-hacking while still allowing exploratory reasoning leaps.

2. Is there a provable relationship between frozen parameter subspaces (scalars \(\alpha_l\)) and the model’s few-shot generalization, or is the trade-off purely empirical? A theoretical treatment would let us design scalars that adapt based on the Jacobian norm of each layer rather than hand-picked heuristics.

3. How do activation sketches interact with distributional shifts in reinforcement learning datasets? BASIS uses invariant scalars defined under the train distribution—what happens when prompts at inference time lie far outside that manifold, and can the sketch be extended to a non-stationary catalyst that re-estimates invariants online?

4. Can prototype-based regularizers like DDCL-INCRT explain why some representations remain stable under RLHF while others collapse? If so, can we construct a falsifiable probe that measures prototype assignment entropy and correlates it with factuality scores?

## Where to read next

If you want the probabilistic backbone of these constraints, → [Policy gradient](../../06-reinforcement-learning/concepts/policy-gradient.md) develops the support for general regularization terms in the objective; the engineering counterpart is → [[rlhf]] which ties those constraints to deployed instruction-tuning pipelines; for a deeper take on frozen subspaces and their trade-offs, → [[parameter-freezing]] sketches how layer-wise scalars control the forgetting curve.

## Build it

Fine-tuning a Qwen-2.5-0.5B policy with a toy GRPO step makes the KL penalty tangible, because you can plot the score of the KL term alongside the reward and see the model’s output distribution drift when \(\beta\) is too small.

**What you're building:** a PyTorch implementation of a KL-regularized GRPO step on Qwen-2.5-0.5B, along with plots that show how varying \(\beta\) alters the divergence between the fine-tuned and reference policies.

**Why this is valuable:** it demonstrates that regularization controls not just loss curves but the actual probability mass of outputs, giving intuition for why RLHF systems rely on \(\beta\) tuning to preserve factuality.

**Stack:**
- **Model:** [Qwen-2.5-0.5B](https://huggingface.co/qwen/Qwen-2.5-0.5B) — real model card with millions of downloads
- **Dataset:** [math_dataset](https://huggingface.co/datasets/math_dataset) — subset of reasoning prompts sized for Colab
- **Framework:** PyTorch 2.2 + Transformers 4.40 + Accelerate 0.20 (from huggingface.co)
- **Compute:** Colab T4 (16GB VRAM) / ~2 hours for a single sweep over 500 prompts

**The recipe:**
1. `pip install torch==2.2.0 transformers==4.40.0 accelerate==0.20.0 matplotlib` and load the Qwen-2.5-0.5B reference weights with `AutoModelForCausalLM`.
2. Tokenize the math prompts, sample 64-token completions, and build \( \pi_{\mathrm{ref}} \) outputs by running the model with `torch.no_grad()`; store their logits for KL computation.
3. Define the loss \(L(\theta)\) as above, compute \(\hat{A}(s,a)\) from a synthetic reward (e.g., length-normalized log-likelihood of a target answer), and backpropagate with \(\beta \in \{0.0, 0.1, 0.5\}\); expect the KL term to dominate when \(\beta\) increases and the advantage to flatten.
4. After each step, compute \(D_{\mathrm{KL}}(\pi_\theta || \pi_{\mathrm{ref}})\) and plot it against \(\beta\) and rollout reward; you should observe that reward increases at \(\beta=0\) but KL explodes, whereas \(\beta=0.5\) keeps KL steady while reward grows slower.
5. The artifact is the model checkpoint plus plots showing KL vs. reward vs. \(\beta\); this artifact captures exactly how regularization keeps the policy anchored.

**Expected outcome:** a checkpointed Qwen-2.5-0.5B policy, plotted curves of KL vs. reward across \(\beta\) values, and a short note on the operating point that best trades off knowledge preservation with task adaptation.

- **CS student:** Run the same recipe on a smaller Qwen-1.0B or even Llama-2-7B for 30 minutes, and replace the KL plot with an entropy plot to keep the computation in the 8GB RAM budget.
- **Applied engineer:** Use the checkpoint to build a quantized vLLM endpoint, serve completions at p50 < 1.2s on an L4, and keep the KL penalty active during inference-time rejection sampling to prevent distributional drift.
- **Applied researcher:** Hypothesize that dynamic \(\beta\) based on KL-aware advantage ratios beats static \(\beta\); implement a controller that raises \(\beta\) when KL grows faster than the reward and compare resulting factuality metrics.
- **Frontier researcher:** Probe the open question about real-time \(\beta\) scheduling (from §What's still open) by measuring how a KL controller tied to prototype assignment entropy (from DDCL-INCRT) affects both reward and factuality curves.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*