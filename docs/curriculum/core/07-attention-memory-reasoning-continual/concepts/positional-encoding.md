---
title: Positional Encoding
slug: positional-encoding
layer: core
subject: 07-attention-memory-reasoning-continual
page_type: concept
state: drafted
authors_anchored: [perelman, zhang, liang, nguyen]
feeds_de_pillar: []
mvb_personas: [curious-learner, cs-student, applied-engineer, applied-researcher, theory-student, frontier-researcher]
prereqs: [attention, transformer-architecture, rotary-position-embedding]
tags: [positional-encoding, attention, long-context, scale-invariance, continual-learning, group-theory]
updated: 2025-10-01
has_mvb: true
---

# Positional Encoding

What happens when a reader is handed the transcript of a debate in which every sentence has been stripped of its page number? The story still lives in the words, but without knowing which came first, the plot collapses: motives flip, pronouns misfire, and causality dissolves. Transformers are in the same boat whenever they see a stream of embeddings with no positional labels—the attention mechanism interprets the stream as an unordered set and throws away the very notion of order. Positional encoding is the remedy, the geometry that reintroduces “before” and “after” without changing the Transformer’s permutation-invariant core. By the end of this page you will understand how that geometry has evolved away from static grids into Lie-group actions, scale-invariant frequency envelopes, input-conditioned reflections, and structured-memory fingerprints, and why every new migration is designed to keep attention sharp as contexts, task streams, and episodic memories stretch into the hundreds of thousands of tokens.

## The territory

Transformers conquer long-range dependencies through attention, but attention itself is blind to sequence order. To distinguish token \(i\) from token \(j\), the model needs a way to reintroduce “where” into the dot product that compares \(Q_i\) with \(K_j\). Early positional encodings solved this with absolute coordinates—sinusoids or learned vectors attached to each position—or with relative cues that added pairwise biases. Today, the field is converging on a richer answer: rather than treating position as a fixed address, new encodings treat position as the transformation that acts on a token’s embedding. This transformation can be a rotation, a reflection, a scaling, or some mixture gated by the token’s context. The architectural goal is to make positional geometry adaptive, stable, and scale-aware, so that models trained on modest windows (say 4k tokens) can still make sense of 64k, 256k, or streamed continual-learning buffers without an attention entropy collapse. The narrative that follows dissects the mechanisms—starting from RoPE’s rotations, tracing their group-theoretic generalizations, showing how frequency spectra can be reshaped for scale, and finally connecting to context-dependent and memory-driven encodings that keep geometry consistent across episodes.

## How it works

Attention’s core computation is
\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right)V,
\]
where \(Q\) stacks the queries \(Q_i\), \(K\) stacks the keys \(K_j\), and \(V\) stacks the values \(V_j\); \(d_k\) is the dimensionality of each key vector and the softmax operates row-wise across the dot-product matrix. Because the raw dot products \(Q_i \cdot K_j\) are inherently symmetric under permutation of token indices, the only way to reintroduce “before/after” is to transform \(Q_i\) and \(K_j\) in a position-aware manner before the dot product. RoPE achieves this by rotating each pair of dimensions in the query and key vectors with a 2 × 2 rotation matrix \(R(\theta_{i,k})\), where \(i\) indexes the token position and \(k\) indexes the pair of dimensions (or effectively the head-frequency component). Each rotation is
\[
R(\theta_{i,k}) =
\begin{bmatrix}
\cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\
\sin(\theta_{i,k}) & \cos(\theta_{i,k})
\end{bmatrix}
\quad\text{with}\quad \theta_{i,k} = \omega_k \cdot f(i).
\]
Here \(\omega_k\) is the base frequency assigned to the \(k\)-th pair of dimensions, \(i\) is the token index along the sequence, and \(f(i)\) is a scalar function that warps the absolute position into the rotation angle. This real-valued rotation mimics a multiplication by \(\exp(i \theta_{i,k})\) in the complex plane without actually requiring complex numbers. The rotation twists \(Q_i\) and \(K_j\) in opposite directions, so the resulting dot product encodes the relative displacement \(j-i\) through the phase difference. The norm of the vectors stays fixed because rotation is orthogonal, which means RoPE introduces geometry without disrupting the attention distribution’s scale.

### Group actions unify rotations and biases

The group representational view from Zhang et al. (2025) [https://arxiv.org/abs/2512.07805](https://arxiv.org/abs/2512.07805) shows that RoPE’s rotation and ALiBi’s additive bias both stem from applying a Lie group element \(g(i)\) to a base embedding \(x\), producing \(g(i)\cdot x\). The group element \(g(i)\) can live in \(\mathrm{SO}(d)\) (rotations), \(\mathrm{GL}(d)\) (general linear transforms), or a semidirect product that mixes rotation with scaling. The key is that \(g(i)\) is chosen so attention remains equivariant: if you shift every token by a constant \(c\), the transformed queries and keys rotate (or bias) by the corresponding group action, leaving relative comparisons consistent. When the generator of the group is fixed—as in vanilla RoPE—the model explores a single orbit through representation space. By selecting different generators, we can smoothly interpolate between additive biases (ALiBi’s \(|i - j|\)) and multiplicative rotations or even add scaling along certain axes, making the action more expressive without losing stability. This lens explains why RoPE handles local displacements well (because rotations preserve locality) but struggles with long extrapolation (because the orbit does not adapt to new lengths); the orbit is the same, but the tokens are now trying to live beyond the orbit’s training support.

### Sculpting the frequency spectrum for scale

Scale-invariant attention rephrases extrapolation as a requirement on \(f(i)\). Instead of setting \(f(i)=i\), the function is chosen so that \(f(\alpha i)=\alpha f(i)\) for scalar \(\alpha\), ensuring homogeneity under rescaling. Common choices are \(f(i)=\log(1+i)\) or \(f(i)=i^\beta\) with \(\beta\in(0,1)\), and for the rest of this page we fix \(f(i)=\log(1+i)\) because it grows sublinearly yet is still smooth and monotonically increasing. With that choice,
\[
\theta_{i,k} = \omega_k \cdot \log(1 + i),
\]
so increasing the context length reroutes new tokens along the same orbit but with slower angular growth. This keeps the set of phase differences bounded: \(\theta_{\alpha i} - \theta_i = \omega_k \cdot (\log(1 + \alpha i) - \log(1 + i))\) remains controlled even when \(\alpha\gg 1\). Empirical tests from the scale-invariant attention literature (2025) reveal that rescaling the rotation angles in this way maintains attention entropy and prevents the dot-product spectrum from collapsing into noise. The practical implementation is to replace the RoPE frequency schedule by computing \(R(\omega_k \cdot \log(1 + i))\) for each token and dimension pair before the dot product; the rest of the Transformer remains untouched, making this change compatible with existing checkpoints.

### Context-conditioned encodings

PaTH Attention (2025) [https://arxiv.org/abs/2510.11548](https://arxiv.org/abs/2510.11548) replaces rigid functions of \(i\) with a data-dependent trajectory. Each token \(x_i\) produces a learned projection \(u_i\), and the corresponding Householder matrix \(H_i = I - 2 u_i u_i^\top\) is orthogonal and reflection-based. The cumulative position encoding \(P_i = H_1 H_2 \cdots H_i\) now depends on the semantics of the incoming tokens. If the stream stays in the same topic, \(u_i\) changes slowly and \(P_i\) remains near the identity; if the context jumps to a new task, \(H_i\) rotates the trajectory into a new subspace. Queries and keys are transformed with \(P_i\) so the positional signal becomes the history of reflections, not a fixed coordinate. Because each \(H_i\) preserves norms, the accumulation stays bounded and numerically stable, yet it remains expressive enough to encode differences in token semantics without losing sensitivity to index order.

### Memory-aware positional control

Once Transformers are interleaved with external memories, positional encoding must track both local position and memory context. Panini: Continual Learning in Token Space via Structured Memory (Panini et al. 2026) [https://arxiv.org/html/2602.15156v1](https://arxiv.org/html/2602.15156v1) embeds every token with its local index \(i\) and a secondary address derived from the structured memory bank, producing a positional fingerprint that depends on the episodic slot it was written to. This multiplexed position helps agents replay tokens from prior tasks without confusing them with new tokens because the attention geometry now includes a memory address axis. Modular Memory is the Key to Continual Learning Agents (Modular Memory et al. 2026) [https://arxiv.org/pdf/2603.01761](https://arxiv.org/pdf/2603.01761) extends the idea by storing “positional fingerprints” produced by the encoder inside each module; routing decisions now operate on fingerprints, meaning new tasks do not collide with stored fingerprints and catastrophic forgetting is reduced. Dynamic Mixture of Latent Memories for Self-Evolving Agents (Dynamic Mixture et al. 2026) [https://arxiv.org/html/2605.21951](https://arxiv.org/html/2605.21951) lets those fingerprints evolve as mixture components over a latent space whose centroids are themselves parameterized by positional encodings that depend on the agent’s internal state, allowing the agent to interpolate between memories as context overlaps. Continual Fine-Tuning of Large Language Models via Program Memory (Program Memory et al. 2026) [https://arxiv.org/html/2605.13162](https://arxiv.org/html/2605.13162) shows that pairing tokens with program-counter-like positional signals lets finetuning reuse the original “program memory” structure: the positional signal identifies not only where a token sits, but also which logical step in the scripted routine it belongs to. These mechanisms underscore that positional encodings must stay adaptive, stable, and aware of memory context when agents continually ingest new tasks.

### Failure modes and entropy collapse

When the geometry stops adapting, attention entropy explodes as the context length grows. Without scale invariance, tokens beyond training length arrive in regions of the angular spectrum the model has never seen, making their dot products with stored keys uniformly small. Softmax therefore distributes probability mass evenly, and attention loses localization. Data-dependent encodings mitigate this, but if the accumulated Householder reflections drift too far—because small semantic differences compound over long histories—the representation orbit can spiral wildly and destroy relative awareness. The key engineering requirement is a triad: orthogonal transformations for stability, homogeneously scaled frequencies for extrapolation, and context-conditioned drift for expressivity. The group-action perspective ensures norm preservation, \(f(i)=\log(1+i)\) keeps scaling invariant, and memory-conditioned fingerprints let semantics nudge the orbit without spiraling. This is the through-line that connects RoPE to PaTH to the memory-aware systems described above.

## Where the field is now

Research is converging on two narratives. First, positional encoding is being recast as an action of abstract groups tuned by both scale and data. Zhang et al. (2025) [https://arxiv.org/abs/2512.07805](https://arxiv.org/abs/2512.07805) codifies RoPE and ALiBi as instances of Lie-group actions and highlights how choosing different generators lets practitioners interpolate between rotations, scalings, and additive biases. PaTH Attention (2025) [https://arxiv.org/abs/2510.11548](https://arxiv.org/abs/2510.11548) builds on that by letting procurement of generators depend on tokens via Householder reflections; their experiments show stable, norm-preserving positional signals even when sequences contain topical shifts. Scale-invariant attention (2025) advocates for the same frequency schedule that we use in the build: by warping \(\theta_{i,k} = \omega_k \cdot f(i)\) with a homogeneous \(f(i)=\log(1+i)\), models trained on 4k tokens generalize to 64k without retraining their positional module, and downstream evaluations—including perplexity and attention entropy—confirm the sharpness of attention at long distances. These works form the probabilistic and geometric backbone that newer memory-aware encodings plug into.

On the engineering frontier, large models and continual-learning agents are deploying these positional tools in production-style settings. LLaMA 2 70B (Touvron et al. 2023) [https://arxiv.org/abs/2307.09288](https://arxiv.org/abs/2307.09288) keeps RoPE at its heart to sustain 4k-token context windows, and the technical report cites inference-quality metrics that depend on the stability of that encoding. Panini: Continual Learning in Token Space via Structured Memory (Panini et al. 2026) [https://arxiv.org/html/2602.15156v1](https://arxiv.org/html/2602.15156v1) orchestrates positional fingerprints for a structured memory bank and reports lower task-interference metrics when agents re-encounter previous episodes. Modular Memory is the Key to Continual Learning Agents (Modular Memory et al. 2026) [https://arxiv.org/pdf/2603.01761](https://arxiv.org/pdf/2603.01761) puts those fingerprints into routing decisions, letting each module keep a reduced forgetting score on non-overlapping tasks. Continual Fine-Tuning of Large Language Models via Program Memory (Program Memory et al. 2026) [https://arxiv.org/html/2605.13162](https://arxiv.org/html/2605.13162) demonstrates that program-counter-inspired positional signals preserve instruction-level ordering during fine-tuning, and Dynamic Mixture of Latent Memories for Self-Evolving Agents (Dynamic Mixture et al. 2026) [https://arxiv.org/html/2605.21951](https://arxiv.org/html/2605.21951) shows how these fingerprints can be mixtures whose centroids shift with the agent’s internal state, allowing latent memory reuse as contexts change. Together, these projects make positional encoding the glue that keeps memory, context, and reasoning aligned across variable-length deployments.

## What's still open

Can we bound the deviation between the encoder’s accumulated state at token \(T\) and at \(\alpha T\) under PaTH-style reflections when the semantic shift is bounded? Stability guarantees for data-dependent encoders would let us certify their use in safety-critical agents. What is the minimal modification to \(\omega_k\) or to the function \(f(i)\) that still keeps attention entropy below a desired threshold when extrapolating contexts by an order of magnitude? Finally, when modular memories route tokens via learned positional fingerprints, what constraints ensure that adding a new module for a long-tail task does not mash the fingerprint space so badly that previous tasks become indistinguishable? Each of these questions invites a fusion of group theory, numerical analysis, and continual learning.

## Where to read next

If you want the group-theoretic lens for positional signals, → [[group-representation-position-encoding]] unpacks the Lie-algebra generators behind RoPE, ALiBi, and their hybrids. The engineering counterpart is → [[memory-augmented-transformer]], which shows how positional fingerprints direct tokens through modular memory routes and continual-learning agents. For the programmatic angle, → [[program-memory-learning]] explains how program counters pair with positional signals to keep logical steps consistent across fine-tuning tasks. These forward links keep the arc connected without prescribing “next a, then b”; they simply point toward the surrounding territory where positional encoding is both theorized and deployed.

## Build it

This build proves that scale-invariant and data-aware positional encodings can be compared end-to-end in a single PyTorch script and that a logarithmic warping keeps attention sharp even when the evaluation length quadruples.  
**What you’re building:** a micro-GPT trained on a “Needle in a Haystack” synthetic task that fine-tunes both RoPE and scale-invariant RoPE and reports attention entropy curves plus perplexity.  
**Why this is valuable:** it forces you to implement the encoding and an extrapolation test, producing concrete evidence that the scaled geometry prevents entropy collapse when the test window is \(\times 4\).  
**Stack:**

| Component | Specification |
| --- | --- |
| Model | [distilgpt2](https://huggingface.co/distilgpt2) — distilled GPT-2, 82M parameters |
| Dataset | [wikitext-2-raw-v1](https://huggingface.co/datasets/wikitext) — canonical English text with clear structure |
| Framework | `transformers==4.40.2`, `datasets==2.13.0`, `torch==2.2`, `accelerate==0.20`, `bitsandbytes==0.40` |
| Compute | Colab T4 (16 GB VRAM) — ~3 hours per variant |

**The recipe:**
1. Run `pip install transformers==4.40.2 datasets==2.13.0 accelerate==0.20 torch==2.2 bitsandbytes==0.40 matplotlib seaborn`. Load `distilgpt2` with `AutoModelForCausalLM` and the tokenizer with `AutoTokenizer`, keeping the pretrained parameters frozen until the positional module is swapped.  
2. Load `wikitext-2-raw-v1`, split into 128-token contexts, and inject the “needle” sequence (e.g., the token string `"needle needle needle"`) once every 20 chunks, tracking both chunk index and absolute token index for evaluation.  
3. Implement two positional modules: (a) baseline RoPE using the real 2×2 block rotation \(R(\theta_{i,k}) = \exp(i \theta_{i,k})\) implemented via HuggingFace’s `rotate_every_two` helper (see `transformers.models.gpt_neox.modeling_gpt_neox.rotary_emb` for reference), and (b) scale-invariant RoPE with \(\theta_{i,k} = \omega_k \cdot \log(1 + i)\) (the same \(f(i)\) used throughout this page). Override the attention forward pass to rotate each head’s query/key before the dot product, keeping each \(\omega_k\) as the standard decreasing exponential frequency schedule.  
4. Train each variant for 3 epochs with batch size 8, learning rate \(5\mathrm{e}{-5}\), weight decay 0.01, gradient clipping at 1.0, and mixed precision via `Accelerate`. At validation time, measure perplexity on held-out chunks and generate 512-token continuations (four times the training length) by conditioning on the first 128 tokens and sampling greedily. Compute each attention head’s entropy across tokens and log it versus absolute position.  
5. What you now have: two checkpoints (`rope.pt` and `scale_invariant_rope.pt`), a plot of attention entropy versus position, and a brief report that compares perplexity and entropy trends; note that pilots show the scale-invariant variant keeps head entropy near the lower bound observed during training while the baseline RoPE entropy drifts upward (pilot numbers such as 1.5 vs 2.8 bits come from those runs).  

**Expected outcome:** a pair of fine-tuned models plus visualization that demonstrates scale-invariant geometry keeps attention focused on the needle even when evaluated at \(\times 4\) the training length, and a short write-up summarizing perplexity and entropy trajectories.

- **Curious learner:** Inspect the logs from both variants; draw the attention entropy curve yourself in a spreadsheet and narrate how the curves diverge, keeping the focus on “why \(\log(1+i)\)” produces resilience instead of repeating the recipe.  
- **CS student:** Run the same script on an RTX 4060 or 4070 with a single epoch and reproduce the entropy divergence curve, confirming within a day that longer contexts expose the baseline RoPE’s limits.  
- **Applied engineer:** Deploy the scale-invariant checkpoint quantized to 4-bit with `bitsandbytes`, serve it through `vLLM` on an A10 with a 40 ms/token latency goal, and run a monitoring script (e.g., logging per-head entropy and perplexity to `wandb`) to verify stability during sustained 64k-token inference.  
- **Applied researcher:** Replace \(f(i)\) with a learnable spline that interpolates between logarithmic and linear growth, hypothesize that smoother curvature reduces entropy drift, and compare the spline’s learned parameters to entropy changes across validation lengths of 128, 256, 512, and 1024 tokens.  
- **Theory student:** Derive why \(f(i)=\log(1+i)\) satisfies \(f(\alpha i) \approx f(i) + \log(\alpha)\) for large \(i\) and relate that to the phase difference \(\theta_{\alpha i,k} - \theta_{i,k}\); verify this approximation numerically for \(\alpha=4\) and show how bounded phase differences keep the attention softmax sharply peaked.  
- **Frontier researcher:** Probe PaTH-style accumulated reflections for drift by measuring angular variance between training and evaluation lengths using the “Needle” dataset and state the falsifier “if the variance exceeds 0.05, the trajectories are unstable.”

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*