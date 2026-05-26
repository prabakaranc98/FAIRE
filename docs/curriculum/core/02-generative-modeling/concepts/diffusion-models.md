---
title: Diffusion Models
slug: diffusion-models
layer: core
subject: 02-generative-modeling
page_type: concept
state: drafted
authors_anchored: [ho, rombach, sohl-dickstein, podell]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [score-matching, variational-autoencoders, unet]
tags: [denoising, score-based, noise-schedule, latent-space, ddpm, production]
updated: 2024-10-05
has_mvb: true
---

# Diffusion Models

Imagine you could take a masterwork, cover it with sand until the outlines disappear, and then teach a neural network to reverse that abrasion grain by grain—restoring the brushstroke, the shading, the intent. Diffusion models are the algorithmic version of that sculpting fantasy: they degrade data into utter randomness through a known process, then learn to undo each tiny step. The trick is that you never ask the network to paint a masterpiece from scratch; instead it learns how to subtract the accumulated “sand” one kick at a time. By the end of this page you will understand why this multi-step choreography lets diffusion outmaneuver adversarial training instabilities, how it is scaled into modern production systems, and what minimal DDPM you can build to see the denoising dance live on MNIST digits.

## The territory

Generative modeling has always balanced two conflicting needs: define a probability distribution broad enough to cover complex natural data, and make sampling from that distribution tractable. Early GANs attacked the first part with a kicker-discriminator pair but buckled on the second, because discriminators were brittle and gradients frequently vanished. Diffusion models answer the same question from the opposite side of the table. Instead of learning to draw samples directly, they define a forward noising process that gradually destroys structure in a series of T small, fixed steps. The resulting chain is a Markov process that leads every data point toward a simple Gaussian, which means the network only ever has to learn how to reverse small, local corruptions. This is why diffusion techniques are often called “score-based” or “denoising” models: the network learns the gradient of the log-density (the score) at each noise level by predicting what noise was added. Because these steps are handcrafted and easy to simulate, diffusion models inherit stability from the forward process while still matching or beating GANs in sample quality. The mechanism is best understood by starting from how the forward corruption is defined and how the network learns to invert it, one timestep at a time.

## How it works

### The forward (corruption) chain

The forward process \(q(x_t \mid x_{t-1})\) is a fixed Gaussian Markov chain that blurs the data sample \(x_0\) with incremental noise, turning structure into entropy. Mathematically,
\[
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I\right),
\]
where \(x_{t-1}\) is the image at timestep \(t-1\), \(\beta_t \in (0,1)\) is the variance schedule for step \(t\), and \(I\) is the identity covariance. Each \(\beta_t\) controls how much of the signal is replaced by noise at that step, and by choosing a small \(\beta_t\) we ensure the chain never jumps too far. Repeated composition yields a closed-form marginal for any \(t\):
\[
q(x_t \mid x_0) = \mathcal{N}\!\left(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t)I\right),
\]
where \(\alpha_t = 1 - \beta_t\) and \(\bar{\alpha}_t = \prod_{s=1}^t \alpha_s\). The remarkable property is that \(x_t\) is a linear interpolation between the original image and Gaussian noise, so sampling \(x_t\) requires just one draw: \(x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon\) with \(\epsilon \sim \mathcal{N}(0, I)\). Sohl-Dickstein et al. (2015) [arxiv:1503.03585](https://arxiv.org/abs/1503.03585) introduced this nonequilibrium thermodynamic view, showing that the forward chain is computationally cheap and that it provides a tractable distribution from which to build the inverse.

### The reverse process and noise prediction

Given the forward chain, we now need a reverse chain \(p_\theta(x_{t-1} \mid x_t)\) that recovers images step by step. Because the forward chain is Gaussian, its reverse can also be modeled as Gaussian:
\[
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)\right),
\]
where \(\mu_\theta\) and \(\Sigma_\theta\) are learned functions. Ho et al. (2020) [arxiv:2006.11239](https://arxiv.org/abs/2006.11239) observed that directly predicting \(\mu_\theta\) is unnecessary if the network instead predicts the noise \(\epsilon\) that was added at each step; the mean can then be computed analytically. The parameterization becomes
\[
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right),
\]
where \(\epsilon_\theta\) is the network output and \(\alpha_t\), \(\bar{\alpha}_t\) are defined as above. This is the key insight that transforms the reverse distribution estimation into a regression problem. The training loss is
\[
L(\theta) = \mathbb{E}_{x_0, t, \epsilon} \left[\left\|\epsilon - \epsilon_\theta\left(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t\right)\right\|^2\right],
\]
where \(x_0 \sim p_{\text{data}}\), \(t\) is drawn uniformly from \(\{1,\dots,T\}\), and \(\epsilon \sim \mathcal{N}(0, I)\). The left-hand side of the loss penalizes differences between the true noise and the model prediction; the right-hand side ensures each timestep gets equal attention. By predicting noise, the network avoids learning the scale of \(x_0\) and focuses on the direction back toward structure—this is why it is described as a denoiser.

Because the variance \(\Sigma_\theta\) can be fixed (e.g., set to \(\beta_t I\)) or even learned slowly, the training objective remains simple and the gradient flow is stable. The loss can also be reweighted to prioritize earlier or later steps, giving a handle on fidelity vs. diversity. Importantly, sampling from the trained reverse chain starts from \(x_T \sim \mathcal{N}(0, I)\) and iteratively applies \(p_\theta(x_{t-1} \mid x_t)\) for \(t = T, \dots, 1\) to reach \(x_0\). Because each step is just a single forward pass through \(\epsilon_\theta\), the process is embarrassingly parallelizable and can be run with as few as a dozen timesteps in practice, albeit with a quality trade-off.

### The U-Net denoiser architecture

The architecture of \(\epsilon_\theta\) mirrors U-Nets that work well on segmentation tasks: a contracting path aggregates context while an expanding path reconstructs noise predictions at the original resolution. The network absorbs the timestep \(t\) by embedding it with sinusoidal features (as used in Transformers) and projecting them into each convolutional block. Each block consists of a pair of convolutions, normalization, and Swish activations, with residual additions to stabilize gradients. Skip connections between contracting and expanding paths preserve spatial detail that would otherwise vanish under the repeated smoothing of the forward chain. The resulting architecture is lean enough to train on 4–8 GB GPUs but expressive enough to capture the multi-scale structure needed for photorealistic samples.

### Latent diffusion and scaling to high resolution

Pixel-space diffusion at \(512^2\) or higher is costly because each convolutional pass must handle large tensors. Rombach et al. (2022) [arxiv:2112.10752](https://arxiv.org/abs/2112.10752) introduced Latent Diffusion Models (LDMs) to shift most of the computation into a compact latent space. An autoencoder \(E\) maps \(x_0\) into latents \(z_0 = E(x_0)\) with a modest downsampling factor (typically \(4\times\) or \(8\times\)), and the decoder \(D\) reconstructs the image from latents. The diffusion process now operates on \(z_t\) rather than \(x_t\), with noise added in the latent space:
\[
q(z_t \mid z_{t-1}) = \mathcal{N}\!\left(z_t; \sqrt{1 - \beta_t} z_{t-1}, \beta_t I \right).
\]
Because the latents have lower dimensionality, the UNet denoiser has fewer layers and fewer channels, reducing memory bandwidth and enabling training on a single V100 or A5000 with \(\sim 16\) GB VRAM. The decoder then projects the denoised \(z_0\) back to the pixel domain: \(x_0 = D(z_0)\). This latent shift is what allowed diffusion models to scale from \(32 \times 32\) research toys to industrial-grade \(1024 \times 1024\) systems while still preserving the fundamental denoising training objective.

### Failure modes and practical notes

Diffusion models trade sampling speed for stability. Running the reverse chain for \(T=1000\) steps yields high-quality samples but incurs a latency cost linear in \(T\). Accelerations such as DDIM or stochastic sampling heuristics exist, but they often require tuning the noise schedule \(\beta_t\) to maintain quality. A poorly chosen schedule causes the forward chain to either collapse (if \(\beta_t\) grows too fast, eliminating signal prematurely) or stagnate (if \(\beta_t\) is too small, making reverse steps ambiguous). Another practical challenge is classifier-free guidance: during sampling we can bias \(\epsilon_\theta\) toward a conditional prompt by interpolating guided and unguided noise predictions, but the strength of this guidance must be calibrated to avoid mode collapse. Finally, when diffusion is applied autoregressively over time (e.g., for videos), errors accumulate across frames; incorrectly handled drift can break temporal coherence unless corrective mechanisms appear in the sampling loop. Addressing drift without backpropagation through long rollouts is an active frontier.

## Where the field is now

DDPM (Ho et al. 2020) [arxiv:2006.11239](https://arxiv.org/abs/2006.11239) showed that the simple squared-error noise prediction objective could match GAN-level fidelity on \(32 \times 32\) benchmarks, which was the first evidence that diffusion’s multi-step reversal strategy is competitive. Latent Diffusion (Rombach et al. 2022) [arxiv:2112.10752](https://arxiv.org/abs/2112.10752) then demonstrated that the same framework, when shifted to a compressed latent via a pre-trained autoencoder, supports \(1024 \times 1024\) image synthesis with inference costs similar to GANs while keeping the denoising training objective intact. The latest research frontier for denoising models is temporal data: Video Diffusion Models (Ho et al. 2022) [arxiv:2204.02891](https://arxiv.org/abs/2204.02891) embed diffusion in an autoregressive framework that enforces consistency across frames, achieving lower FVD (Fréchet Video Distance) than previous GAN-based video generators for 64-frame clips.

Engineering-wise, Latent Diffusion powers Stability AI’s Stable Diffusion 2.1 and its downstream services, where the latent-space denoiser runs on billions of API calls per month while consuming under 8 GB of VRAM per inference, according to Stability AI’s engineering blog [stability.ai/blog/stable-diffusion-2-1](https://stability.ai/blog/stable-diffusion-2-1). The same production stack leverages distilled UNets, 8-bit quantization, and scheduler caching to deliver prompt-to-image latencies around 2 seconds on A100-class hardware. This combination of algorithmic stability (research frontier) and lean inference (engineering frontier) is why diffusion models now appear in creative tools, robotics simulators, and even medical imaging suites.

## What's still open

Can we remove the dependence on a long reverse chain by learning a single-step denoiser whose predictions correct the residual discovered after running a vanilla DDPM sampler for \(k\) steps? If such a correction exists, it would replace expensive sampler distillation while still maintaining fidelity. What is the minimal augmentation to the noise schedule that makes timestep-wise guidance calibration generalize across domains so that guidance strengths tuned on one dataset do not fail catastrophically on another? The most immediate open question raised earlier is this: how can autoregressive video diffusion models detect and correct their own accumulated drift errors during inference without the computationally prohibitive cost of backpropagating gradients through long, sequential multi-step rollouts? A solution that treats drift correction as an online filtering or learned controller would unlock generative video that remains coherent for thousands of frames. Each of these questions points to a different weakness of the current denoising pipeline—nonlinear correction, schedule generalization, temporal stabilization—and solving any one would move diffusion modeling from strong offline synthesis toward robust, closed-loop generation.

## Where to read next

If you want the probabilistic foundation, → [[score-matching]] exposes the score estimation view that DDPM compiles down to; the engineering counterpart for making these UNets efficient is → [[flash-attention]], which explains how attention-heavy decoders run on commodity GPUs; for operating on compressed latents, → [[latent-diffusion]] walks through the encoder-decoder pair that lets diffusion handle high resolutions; and the next paradigm beyond diffusion is → [[flow-matching]], which replaces the discrete timesteps with continuous paths that can be solved with single backward passes.

## Build it

A minimal DDPM trains the denoiser on MNIST digits so you can see the erosion-reversal loop turn random noise into recognizable handwriting in under ten minutes. This build proves the core insight that learning to subtract incremental Gaussian noise, rather than directly predicting pixels, makes sampling stable even on commodity hardware.

**What you're building:** A DDPM trained from scratch with a tiny U-Net denoiser that generates \(28 \times 28\) MNIST digits from random Gaussian noise on a 10-minute Colab T4 run.

**Why this is valuable:** It forces you to implement the forward noising schedule, the noise-predicting loss, and the reverse sampling loop, which together encode the mathematical core that differentiates diffusion from GAN-style generation.

**Stack:**
- **Model:** `google/ddpm-cifar10-32` [https://huggingface.co/google/ddpm-cifar10-32] — >4.5K downloads; use its `UNet2DModel` config as the blueprint for your MNIST denoiser.
- **Dataset:** `mnist` [https://huggingface.co/datasets/mnist] — 60K grayscale \(28 \times 28\) digits, already normalized.
- **Framework:** PyTorch 2.1 + `diffusers` 0.30 (for the scheduler and UNet boilerplate).
- **Compute:** Colab T4 (16 GB VRAM) — entire training finishes in ~8 minutes with 1000 steps; sampling another 15 seconds per checkpoint.

**The recipe:**
1. Install + load: `pip install torch torchvision diffusers accelerate datasets` and import `UNet2DModel`, `DDPMScheduler`, and `MNIST` from the Hugging Face ecosystem; set `torch.backends.cudnn.benchmark = True`.
2. Data: normalize MNIST to \([-1,1]\), stack into batches of 128, and use random horizontal flips only as augmentation to keep the digits centered; the scheduler expects tensors shaped `[B, 1, 28, 28]`.
3. Train/fine-tune: instantiate a `DDPMScheduler` with \(T=1000\) steps and \(\beta_t\) linearly spaced from 0.0001 to 0.02; run 5 epochs with AdamW at lr=1e-4, weight decay 0.01, gradient norm clipping 1.0, and the loss \( \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \). Expect the training loss to drop below 0.5 after ~3 epochs on the T4.
4. Evaluate: sample 8 batches of 64 images by starting from \(\mathcal{N}(0, I)\) at \(t=1000\) and reversing with the learned scheduler; the FID against the MNIST test set should stabilize around 7.5 (it is easy to compute with `torchmetrics`). Save a grid of generated digits.
5. What you now have: a checkpointed denoiser and sampling notebook that turn Gaussian noise into digits, along with visual proof (saved grids) that the reverse chain mirrors the forward erosion.

**Expected outcome:** A training checkpoint plus a visualization notebook showing MNIST digits emerging step-by-step from noise, ready for further experiments or export to other datasets.

- **CS student:** Use a free Colab T4 and reduce \(T\) to 500 steps; lower batch size to 64, and log the per-step reconstruction loss so you understand the stability of the noise-prediction objective.
- **Applied engineer:** Export the trained UNet to ONNX, quantize it to 8-bit with Hugging Face Optimum, and serve it through a `diffusers` pipeline behind a vLLM-style HTTP endpoint targeting <250 ms inference on an A10.
- **Applied researcher:** Hypothesize that a cosine \(\beta_t\) schedule improves fast sampling; train with both linear and cosine schedules, hold all else constant, and compare FID after 250 sampling steps to test the hypothesis.
- **Frontier researcher:** Probe the open question about autoregressive video drift by fine-tuning this MNIST DDPM to denoise short MNIST-video clips and measure how per-frame drift accumulates when you skip gradient updates through the sequence, aiming to falsify the idea that independent timestep correction suffices.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*