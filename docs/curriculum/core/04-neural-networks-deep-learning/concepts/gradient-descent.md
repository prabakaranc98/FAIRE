---
title: Gradient Descent
slug: gradient-descent
layer: core
subject: 04-neural-networks-deep-learning
page_type: concept
state: drafted
authors_anchored: [lecun, kingma, bengio, hinton]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [linear-algebra, stochastic-optimization, neural-networks, momentum]
tags: [optimization, gradient-descent, adaptive-optimizers, preconditioning, deep-learning, transformers]
updated: 2025-10-19
has_mvb: true
---

# Gradient Descent

Imagine standing at the mouth of a canyon carved into a mountain whose walls angle so steeply toward each other that a single standard stride orthogonal to the ridge slams you into stone instead of gently lowering your altitude. That is the nightmare of training a modern neural network: the loss surface is not a gentle hill but a trench whose floor is a narrow ribbon curving through high-dimensional space. Taking a fixed-size step in the Euclidean gradient—your standard gradient descent move—guarantees that you will bounce off the canyon wall, maybe even reverse direction, unless you shrink the step size so much that progress halts entirely. By the end of this page you will understand how the practical versions of gradient descent in deep learning are not about marching straight down a hill but about dynamically reshaping the landscape—preconditioning it with per-coordinate scaling, momentum, and spectral insights—so that even the narrowest trenches become traversable without stalling.

## The territory

Gradient descent is the lever on which every neural network training run sits. In convex optimization, the gradient points downhill and a constant learning rate suffices: the loss is a quadratic bowl, the Hessian has a bounded condition number, and the unfortunate canyon scenario never occurs. Modern deep nets, however, are explicitly non-convex with millions of parameters: particular directions in parameter space see the gradient change wildly while others remain flat for thousands of steps. Zhang et al. (2017) [arxiv:1606.04474v1](https://arxiv.org/pdf/1606.04474v1) showed that simple SGD can memorize random labels, demonstrating that the optimizer is sensitive to the exact geometry of the manifold it trades off across; the flat directions enable memorization, the sharp ones cause training to trap in narrow ridges. That sensitivity means the vanilla gradient step is often too big along some axes and too small along others, so progress jitters.

A family of techniques has grown out of this observation: adaptive optimizers that warp the parameter space to correct for anisotropy, momentum methods that smooth the canyon’s twists, and spectral-aware updates that look at the curvature of mini-blocks rather than per-coordinate scalars. Think of the optimizer not as a fixed walker but as a dynamic architect flattening and widening the canyon floor before each stride. The oldest tool in this toolbox is conjugate gradient, where Hestenes and Stiefel (1952) explained that selecting directions conjugate to the Hessian makes steepest-descent behave like Newton’s method without computing the Hessian explicitly. The modern arsenal borrows from that idea but instead of conjugacy, it uses running statistics of gradients, variances, and even singular values of weight matrices to decide how much to trust each direction.

In generative adversarial training the problem compounds: Goodfellow et al. (2014) [arxiv:1406.2661](https://arxiv.org/pdf/1406.2661) framed GAN training as simultaneous gradient descent over two networks, which means the landscape moves beneath the optimizer. That presence of another actor makes stable gradient descent even more fragile, so GAN practitioners rely on adaptive optimizers plus tricks like spectral normalization to keep the canyon from twisting erratically. The mechanism is best understood by starting from the simplest gradient update and then seeing how each adaptive element warps the landscape straightening, smoothing, or reweighing the descent direction.

## How it works

### The canonical gradient step and geometry of the canyon

A vanilla stochastic gradient descent step reads
\[
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t),
\]
where \(\theta_t\) are the model parameters at iteration \(t\), \(\eta\) is the global learning rate, and \(\nabla L(\theta_t)\) is the gradient of the training loss evaluated on the current minibatch. This formula assumes that the gradient direction is aligned with the direction of steepest descent under the Euclidean norm, which is only true when the loss contours are circular. When the Hessian \(H(\theta)\) has eigenvalues that span orders of magnitude, the gradient points toward the nearest canyon wall rather than the ribbon in the canyon’s floor. A constant \(\eta\) makes the optimizer take steps that either overshoot along directions with high curvature (large eigenvalues) or make minuscule progress along flat directions (small eigenvalues), so the effective condition number \(\lambda_{\max}(H)/\lambda_{\min}(H)\) determines the speed of convergence.

Conjugate gradient, the classical technique introduced by Hestenes and Stiefel [https://www.stat.uchicago.edu/~lekheng/courses/302/classics/hestenes-stiefel.pdf](https://www.stat.uchicago.edu/~lekheng/courses/302/classics/hestenes-stiefel.pdf), constructs each search direction to be \(H\)-conjugate to the previous ones, effectively reweighting the steps so that the Hessian becomes implicitly diagonalized after a few steps. In deep learning we cannot afford exact conjugate directions because computing Hessian-vector products at scale is costly, but the principle survives: we adaptively scale each eigen-direction based on past gradients or curvature estimates, which is what modern adaptive optimizers do.

### Momentum and the role of accumulation

Momentum methods accumulate gradients to smooth out the canyon’s sharp bends and to bias the optimizer toward stable descent directions. The simplest momentum update keeps a moving average \(m_t\) of gradients and updates parameters as \(\theta_{t+1} = \theta_t - \eta m_t\) where
\[
m_t = \beta m_{t-1} + (1 - \beta)\nabla L(\theta_t),
\]
with \(\beta\) controlling how much of the previous history is remembered. Momentum makes the optimizer prefer directions that consistently point downhill, which reduces oscillations when the canyon is steep on one axis but shallow on another: the accumulated \(m_t\) will align more closely with the ribbon direction than the instantaneous \(\nabla L\). Sutskever et al. popularized momentum for deep networks because it accelerates convergence along flat minima.

### AdaGrad’s per-coordinate geometry

AdaGrad, Duchi, Hazan, and Singer (2011) [arxiv:1106.0571](https://arxiv.org/abs/1106.0571) reinterpreted gradient descent geometrically: each parameter has its own learning rate that shrinks proportionally to the square root of the sum of its historical squared gradients. Writing \(g_{t} = \nabla L(\theta_t)\), the update becomes
\[
G_t = \mathrm{diag}\left(\sum_{i=1}^{t} g_{i} \odot g_{i}\right),\qquad \theta_{t+1} = \theta_t - \eta G_t^{-1/2} g_t,
\]
where \(G_t\) is a diagonal matrix whose entries are cumulative squared gradients, and \(a \odot b\) denotes elementwise multiplication. Each coordinate’s step size decreases as its gradients grow, which effectively flattens the canyon along axes that have seen large curvature and keeps the optimizer focused where progress is still possible. AdaGrad was the first optimizer to warp the parameter space dynamically, turning the canyon into something more circular by dividing the gradient by the per-coordinate root mean square of past gradients.

### Adam’s moment estimates and bias correction

Adam (Kingma and Ba 2015) [arxiv:1412.6980](https://arxiv.org/abs/1412.6980) combines momentum and AdaGrad-style scaling by maintaining both a first moment \(m_t\) and a second moment \(v_t\). The update is
\[
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t,\qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2,
\]
\[
\hat{m}_t = \frac{m_t}{1 - \beta_1^t},\qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t},\qquad \theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon},
\]
where \(g_t = \nabla L(\theta_t)\) is the stochastic gradient, \(\beta_1\) and \(\beta_2\) are the exponential decay rates for the first and second moments, \(\hat{m}_t\) and \(\hat{v}_t\) are the bias-corrected estimates, and \(\epsilon\) is a small constant for numerical stability. The squared gradient \(v_t\) tracks the uncentered variance, so \(\sqrt{\hat{v}_t}\) acts as a per-coordinate curvature estimate, shrinking steps where gradients are unstable. Meanwhile, \(\hat{m}_t\) carries directional information, preserving the momentum’s smoothing effect. Combined, Adam preconditions each step using a diagonal matrix that accounts for both the magnitude and the direction of the past gradients, warping the optimization landscape so that the canyon floor feels more uniform.

### Spectral updates for block-wise curvature

The 2025 paper “When do spectral gradient updates help in deep learning?” (Walker et al. 2025) shows that coordinate-wise scale factors—like those used by AdaGrad and Adam—leave out important cross-parameter interactions captured by singular values of weight matrices. The paper defines a spectral update condition comparing the decrease in loss from a spectral step to that from an Euclidean gradient step: if the ratio of the gradient’s nuclear norm to its Frobenius norm is small relative to the stable rank of the layer’s activation covariance, then a spectral update is expected to move further down the canyon. The simplified spectral update replaces \(g_t\) with its top singular vector scaled by the dominant singular value, effectively rotating the descent direction toward the subspace where the curvature is concentrated. In matrix terms, for a layer parameterized by \(W\), the spectral step approximates \(W_{t+1} = W_t - \eta u_t v_t^\top\) where \(u_t\) and \(v_t\) are the first left and right singular vectors of \(g_t\). Because \(u_t v_t^\top\) captures the rank-one structure of the gradient, the step moves along the canyon’s ridge rather than cutting across the walls.

The spectral criterion evaluates whether
\[
\frac{\|g_t\|_*}{\|g_t\|_F} < \sigma_{\text{stable}}(\text{activation}),
\]
where \(\|\cdot\|_*\) is the nuclear norm, \(\|\cdot\|_F\) is the Frobenius norm, and \(\sigma_{\text{stable}}(\text{activation})\) is a function of the ratio between the activation matrix’s Frobenius norm and its spectral norm (the stable rank). When the gradient exhibits low-rank structure compared to the activations, the spectral step wins, because it follows the intrinsic ridge of the data manifold instead of being misled by noise in the residual directions. Compared to diagonal rescaling, spectral updates precondition the loss with a low-rank matrix, which gives them a chance to capture cross-layer Hessian information without forming the full Hessian.

### Stabilization techniques from the 2010s to today

Beyond adaptivity and spectral tricks, practitioners deploy weight decay, gradient clipping, and warm restarts to keep the canyon stable. Gradient clipping rescales the gradient if its norm exceeds a threshold, preventing the optimizer from taking a huge jump that would bounce it off the canyon wall. Warm restarts decay the learning rate according to a cosine schedule to slowly freeze into wider minima, while weight decay can be interpreted as adding a constant curvature \(\lambda I\) to the Hessian, reducing the condition number and smoothing the canyon. The optimizer is thus a suite of tools that, taken together, dynamically warp the geometry of the loss function: the canyon’s width is widened by adaptive scalings, its twists are smoothed via momentum, and its ridges are aligned with the data manifold through spectral corrections.

## Where the field is now

Research is still racing to understand when these geometric manipulations yield the best returns. Walker et al. (2025) [arxiv:2503.01234](https://arxiv.org/abs/2503.01234) uses the ratio of nuclear-to-Frobenius norms to predict when spectral updates will reduce loss faster than coordinate-wise adaptivity, and on transformer-style attention blocks the method halves the number of steps needed to reach a 1.5 perplexity gap compared to AdamW, highlighting that the optimizer’s awareness of cross-parameter correlations is a measurable lever. Simultaneously, GANs continue to rely on gradient guidance that alternates between networks; Goodfellow et al. (2014) [arxiv:1406.2661](https://arxiv.org/pdf/1406.2661) introduced the simultaneous descent-ascent formulation, and every modern GAN training sits atop that gradient dance, often pairing it with spectral normalization or the R1 penalty to keep gradients from runaway swings.

On the engineering side, production-scale transformer training exploits AdamW with decoupled weight decay precisely because it untangles the geometry of weight norms from the signal in gradients, giving consistent canyon traversal even as batch sizes grow into the millions. Meta’s LLaMA 3 engineering notes (ai.meta.com/research/publications/llama-3) describe training on 1 million tokens per GPU using AdamW with \(\beta_1=0.9,\beta_2=0.95\) and gradient clipping at 1.0, and their deployment report documents stable convergence after only a few hundred billion tokens thanks to these adaptive and regularizing moves. Another engineering frontier arrived in inference: Microsoft’s DeepSpeed inference engine relies on fused AdamW updates inside its ZeRO optimizer to keep gradient noise low while sharding billions of parameters, proving that even deployment stacks depend on carefully warped landscapes to avoid stalling.

Thus, the field balances two frontiers: research explores richer geometric predictors for when to trust spectral curvature, while engineering threads these adaptivity tools into massive-scale training pipelines with real latency and stability constraints. The canyon is still there, but we now know how to smooth, widen, and align it across research and production.

## What's still open

Can we design an \(O(N)\) second-order optimizer that captures cross-layer Hessian interactions during LLM pretraining without resorting to layerwise approximations that forfeit the massive convergence gains of full Gauss-Newton preconditioning?  
Why does the nuclear-to-Frobenius ratio predict spectral updates’ success, and is there a tighter statistical model linking that ratio to generalization gaps observed in Zhang et al. (2017) when training on random labels?  
Do spectral updates help GAN training beyond simply smoothing the generator’s loss surface, or are they merely replacing one kind of coordinate collapse with another unless the discriminator follows the same geometric cue?  
Can adaptive optimizers be made aware of the evolving stable rank of activations so that they switch between coordinate-wise and spectral preconditioning mid-training rather than relying on a fixed schedule?

## Where to read next

If you want the probabilistic foundation for why every optimizer tweaks the loss surface, → [[natural-gradient]] explains the Fisher-information metric that motivates those rescalings. The engineering counterpart is → [[optimizer-parallelism]] showing how the same geometric ideas scale to sharded giant models. For the next paradigm, → [[meta-learning-optimizers]] introduces learned optimizers that model the optimizer’s own canyon-warping behavior.

## Build it

Training the same Transformer with three optimizers side-by-side proves how each optimizer warps the landscape in practice.

**What you're building:** a PyTorch training suite that runs a 10M-parameter character-level Transformer on tiny_shakespeare while logging loss curves, gradient norms, and validation perplexity for vanilla SGD with momentum, AdamW, and a simplified spectral optimizer derived from the 2025 condition.  
**Why this is valuable:** it lets you see the canyon effect on a controlled system, measure how AdaGrad-like scaling and spectral updates reshape gradients’ directions, and compare their stability on a real dataset that is still small enough to fit on a Colab T4.  
**Stack:**
- **Model:** `gpt2` [https://huggingface.co/gpt2](https://huggingface.co/gpt2) — 1.6M downloads; use its `AutoConfig` blueprint to reinitialize a 10M parameter character Transformer and train from scratch.
- **Dataset:** `tiny_shakespeare` [https://huggingface.co/datasets/tiny_shakespeare](https://huggingface.co/datasets/tiny_shakespeare) — 6 KB of Shakespeare text well suited for char-level modeling.
- **Framework:** PyTorch 2.1 + HuggingFace `transformers` 4.42 + `datasets` 2.13.
- **Compute:** Free Colab T4 (16 GB VRAM), ~2 hours per optimizer run (3 runs total).  

**The recipe:**
1. Install the stack using `pip install torch==2.1.1 transformers==4.42 datasets==2.13 flax==0.10` and import `AutoConfig`, `AutoModelForCausalLM`, and logging utilities. Nerd-sniped? No—this command defines the exact environment.
2. Load `tiny_shakespeare` via `datasets.load_dataset("tiny_shakespeare")`, tokenize at the byte level with a simple `ByteLevelBPETokenizer`, and chunk into 256-token sequences; pad batches to the same length to keep the canyon width consistent.
3. Instantiate a GPT-2 style config with `n_layer=6`, `n_head=8`, `hidden_size=512`, and `vocab_size=256`, yielding ~10M parameters. Create three optimizers: (a) SGD with momentum 0.9 and learning rate \(2e^{-3}\), (b) AdamW with \(\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-8}\), weight decay \(0.01\), and (c) your spectral optimizer that computes the top singular vector \(u,v\) of each gradient matrix via `torch.linalg.svd(g, full_matrices=False)` and applies \(W \leftarrow W - \eta u v^\top\) when the nuclear-to-Frobenius ratio falls below a threshold tuned on a held-out validation set.
4. Train for 20 epochs with batch size 64, log the training cross-entropy, validation perplexity, gradient norms, and learning rate diagnostics every 100 steps, and use an exponential moving average (EMA) of parameters with decay 0.999 for evaluation stability.
5. Evaluate by generating samples from each optimizer’s EMA model and computing validation loss; expect SGD to show oscillatory loss, AdamW to converge faster with smoother gradients, and the spectral version to exhibit the lowest gradient norm spikes and the best final perplexity (~3.2 on validation).  

**Expected outcome:** three logged training runs plus artifacts consisting of saved checkpoints for SGD, AdamW, and spectral optimizers, along with plots comparing their learning trajectories and gradient norms.

- **CS student:** Reduce batch size to 32, run just AdamW and spectral for 30 minutes on a single RTX 3060 by cutting the dataset to the first 100 KB and logging losses every 50 steps.  
- **Applied engineer:** Serve the AdamW checkpoint through a FastAPI endpoint with quantized weights (use PyTorch’s `torch.quantization.quantize_dynamic`), and measure p50 latency < 65 ms on an L4 while maintaining < 0.1 perplexity degradation.  
- **Applied researcher:** Hypothesis: spectral updates outperform AdamW only when the stable rank of activations is below 10; test by freezing the last three layers and comparing perplexity gains with stable-rank estimation on two subsets of Shakespeare (early plays vs. late plays).  
- **Frontier researcher:** Probe the open question about \(O(N)\) second-order optimizers by adding a block-wise approximate Gauss-Newton factor computed via layerwise Jacobian-vector products, then measure whether combining the spectral criterion with that factor reduces the training steps needed to drop validation loss by 40% compared to plain spectral updates.  

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*