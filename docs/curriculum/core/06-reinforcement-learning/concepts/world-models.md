---
title: World models
slug: world-models
layer: core
subject: 06-reinforcement-learning
page_type: concept
state: drafted
authors_anchored: [ha, schmidhuber, yu, richens]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [variational-autoencoders, recurrent-dynamics, model-based-reinforcement-learning]
tags: [world-models, latent-dynamics, dreamer, generative-modeling, policy-optimization, simulation]
updated: 2025-10-07
has_mvb: true
---

# World models

Imagine an agent that never touches the Doom engine but still learns to survive every salvo of monsters. It sits in a dark room, staring at latent representations of the game as its only input, iteratively predicting how the world will evolve for ten imagined seconds, then rehearsing a policy inside that hallucinated trajectory. When it finally goes live, the agent already knows where monsters will appear because it has practiced the same sequence hundreds of times in a compact, latent dream. The leap that world models claim is not a clever simulator; it is the idea that the agent builds an internal predictive engine, a private theory of reality that it can query, debug, and improve without one frame of real interaction until the predictions are trustworthy enough to act on.

## The territory

Reinforcement learning has always faced a data bottleneck: physical or simulated environments are expensive, and every trial gives only one narrow slice of experience. World models sit at the intersection of representation learning, generative modeling, and planning. They ask: instead of rolling out policies by repeatedly stepping the real environment, can we learn a compact latent that captures the statistics of observations and actions, run the policy inside that latent with cheap predictors, and only verify the final policy on the real MDP? That is what Ha et al. (2018) — “World Models” [https://arxiv.org/abs/1803.10122](https://arxiv.org/abs/1803.10122) demonstrated by decoupling the architecture into three components: perception (the variational autoencoder), memory (the recurrent network that unrolls future latents), and the controller that reads the dreamed trajectory to output actions. The territory of world models is thus model-based reinforcement learning, but one where the model is not a white-box simulator of physics—it is a generative model trained end-to-end inside the agent, so it becomes something the agent can interrogate with imagination.

The question that separates world models from generic simulators is whether the learned latent generator is useful for optimization. Untitled (2018) [http://arxiv.org/pdf/1809.01999v1](http://arxiv.org/pdf/1809.01999v1) emphasized that the transition model must be action-conditional and predictive enough to sustain long rollouts; otherwise, policies overfit to unrealistic dreams. Recent work has extended this idea into large-scale agents: Reinforcement World Model Learning for LLM-based Agents (2026) [https://arxiv.org/pdf/2602.05842](https://arxiv.org/pdf/2602.05842) trains language and vision-language models to use world models for planning, showing that even knowledge-intensive agents can treat latent rollouts as a safe playground for reasoning. In this sense, world models answer the problem of data efficiency and safety by giving every agent a private cosmos of imagined outcomes, and the rest of this page explains exactly how that cosmos is built, what it learns, and how to put it to work.

## How it works

Latent-state perception and memory are the bedrock of every world model. The first component is usually a variational autoencoder that compresses high-dimensional observations \(x_t\) (RGB frames, point clouds, tool outputs) into a latent vector \(z_t\). The VAE is trained to minimize a reconstruction cost plus a KL regularizer:

\[
\mathcal{L}_{\text{VAE}}(\phi, \psi) = \mathbb{E}_{x_t \sim \mathcal{D}} \left[ \mathrm{KL}(q_\phi(z_t \mid x_t)\,\|\,p(z_t)) - \mathbb{E}_{z_t \sim q_\phi(z_t \mid x_t)} \log p_\psi(x_t \mid z_t) \right]
\]

where \(q_\phi(z_t \mid x_t)\) is the encoder parameterized by \(\phi\), \(p_\psi(x_t \mid z_t)\) is the decoder parameterized by \(\psi\), \(p(z_t)=\mathcal{N}(0, I)\) is the prior we regularize toward, and the dataset \(\mathcal{D}\) contains observations paired with actions from an exploratory policy. The left term encourages the latents to stay close to the prior, making sampling stable, while the right term is the expected log-likelihood of reconstructing \(x_t\), ensuring the latents keep the details the controller needs.

The second component is the dynamics model. Let the recurrent hidden state be \(h_t\) and the action taken at time \(t\) be \(a_t\). The learned transition is usually expressed as

\[
h_{t+1} = f_\theta(h_t, z_t, a_t), \qquad \hat{z}_{t+1} \sim g_\theta(h_{t+1})
\]

where \(f_\theta\) is an RNN cell (often an LSTM or GRU) that fuses the previous hidden state \(h_t\), the compressed observation \(z_t\), and the action \(a_t\), while \(g_\theta\) decodes the hidden state back to the next latent \(\hat{z}_{t+1}\). The entire transition model is trained to maximize the likelihood of the latent trajectory:

\[
\mathcal{L}_{\text{transition}}(\theta) = -\sum_t \log p_\theta(z_{t+1} \mid h_{t+1})
\]

where \(p_\theta(z_{t+1} \mid h_{t+1})\) is typically Gaussian. In practice, world models use teacher-forcing during training: they feed the true \(z_t\) from the encoder so the RNN learns the correct dynamics in a supervised manner before being asked to dream.

Once VAE and RNN are trained, the agent enters the dream. The policy \(\pi_{\omega}(a_t \mid h_t, z_t)\) is trained inside this latent space by unrolling the dynamics model for a fixed horizon \(H\). Instead of maximizing real reward, the policy maximizes imagined reward collected along the dream trajectory:

\[
\max_\omega \mathbb{E}_{\hat{\tau} \sim \text{dream}(\pi_\omega)} \left[ \sum_{t=0}^{H-1} \gamma^t \hat{r}_t \right]
\]

here \(\hat{\tau}\) denotes the sequence \((z_0, a_0, z_1, a_1, \dots)\) generated by sampling actions from the policy and rolling the RNN, \(\hat{r}_t\) is the reward predictor conditioned on the hidden state \(h_t\) (often a linear layer), and \(\gamma\) is the discount factor. The expectation is over the stochasticity of the dream model (as \(g_\theta\) samples \(z\)) and the policy. Because this rollout is cheap, the controller can be trained with standard RL algorithms like evolution strategies, PPO, or even cross-entropy methods.

Two details keep the dream useful. First, the reward predictor is trained on real environment rewards paired with the same latents, so \(\hat{r}_t\) remains grounded even though it is consumed entirely inside imagination. Second, the controller is evaluated by rolling the dream out and measuring cumulative reward, but every few thousand policy iterations the agent executes the controller in the real environment to gather new data, augmenting \(\mathcal{D}\) and avoiding divergence.

World models therefore generate a dataset of imagined trajectories; the ELBO trains the perception model, the sequence loss trains the transition model, and the policy objective trains the controller. Because everything is differentiable, modern variants often train some components end-to-end, enabling the controller gradients to shape the encoder for task-relevant compressions—a key insight from Untitled (2018) [http://arxiv.org/pdf/1809.01999v1](http://arxiv.org/pdf/1809.01999v1), who showed that coupling perception and transition training yields latents that capture controllable factors rather than just reconstructible pixels.

### Imagination with verification

World models are not only a way to reuse data; they are the mechanism that lets agents test hypotheses before acting. Reinforcement World Model Learning for LLM-based Agents (2026) [https://arxiv.org/pdf/2602.05842](https://arxiv.org/pdf/2602.05842) introduces a training loop where large language models and vision-language models alternate between dreaming and execution. The LLMs interpret the latent dreams as narratives and produce action candidates; those actions are evaluated both inside the dream and on a small set of real interactions. Because dreams can be executed thousands of times faster than the real world, the majority of optimization happens within imagination, but the small verification set ensures the predictor stays consistent with physics and semantics.

RLVR-World (Yu et al. 2025) [https://arxiv.org/abs/2505.13934v2](https://arxiv.org/abs/2505.13934v2) takes the same idea further by augmenting the reward predictor with a verifier network that estimates whether a dreamed outcome matches what would happen on the real environment. During RL training, the reward is weighted by the verifier’s confidence, thus encouraging policies that stay within the regime where the world model is accurate. The verifier is trained with contrastive samples: real transitions are positive, imagined transitions with perturbed latent drift are negative. This reinforcement with verifiable rewards combats the compounding errors that would otherwise make long dream rollouts useless.

### Failure modes and mitigation

The central failure mode of world models is drift—after a few imagined steps, the latent state wanders away from the manifold of real observations, generating physically impossible frames that lead the controller to overfit. Drift is particularly devastating when actions are temporally abstract macros because a single macro alters the latent manifold more strongly. Mitigations fall into three families: regularization on the latent norms (e.g., a penalty on \(\|z_t\|\) to keep it near the prior), grounding dreams via real data (every \(K\) imagined trajectories are rolled out in the real simulator), and adversarial verification (the RLVR verifier). In applied settings, drift is the reason world models frequently include a short imagination horizon or use latent dropout to widen the training distribution.

World models can also fail to capture semantics needed for reasoning. When the latent is trained purely on pixel reconstruction, it may ignore high-level objects that are crucial for planning, especially in tasks where reward depends on unseen variables. The WorldPrediction benchmark introduced in 2025 (Richens et al. 2025 — [https://arxiv.org/abs/2506.01622](https://arxiv.org/abs/2506.01622)) highlights this gap by evaluating dreamed trajectories for semantic consistency rather than pixel fidelity: the benchmark exposes that even modern world models can predict plausible pixels that nevertheless contradict the underlying scene graph. Closing that gap requires joint training objectives that measure semantic and temporal alignment, a direction explored by the verifiable rewards in RLVR-World.

## Where the field is now

The present frontier of world models is multi-modal, RL-trained imagination. RLVR-World (Yu et al. 2025) [https://arxiv.org/abs/2505.13934v2](https://arxiv.org/abs/2505.13934v2) achieved a benchmark win by training a single latent world model that handles symbolic actions, vision, and text descriptions. The paper reports that RLVR agents achieve 74% success on the ProcGen suite while using 30× fewer real environment steps, thanks to the verifier-weighted rewards and the policy’s ability to query thousands of imagined rollouts per gradient step. This is the research frontier: combining rigorous RL objectives with verifiable world models to let high-capacity language agents reason about future states while ensuring those states remain physically plausible.

On the engineering side, several teams are already shipping deployed products built on the same philosophy. Hugging Face’s StableToolBench release of mradermacher/WorldModel-Stabletoolbench-Llama3.1-8B-i1-GGUF and its companion mradermacher/WorldModel-Stabletoolbench-Qwen2.5-7B-i1-GGUF serve as inference stacks where the model uses a lightweight world model to predict the consequences of tool chains before executing them. These weights are optimized for low-latency planning with constrained compute, demonstrating that a world-model layer can be inserted between language and tools to catch hallucinations before tool invocation. The deployment story here is that world models become a guardrail: they run thousands of imagined tool sequences, score them via a verifier, and only issue real tool calls when the dream has high confidence, reducing misfire rates in production APIs.

Two frontiers remain. Research needs better benchmarks that measure semantic fidelity rather than just pixel error—the WorldPrediction benchmark is a start, but no community-standard exists for visual-linguistic consistency over 50-step horizons. Engineering needs more efficient verifiers: the current Hugging Face stacks run a separate verification network per dream, which is too slow for multi-tool pipelines. Solving those bottlenecks will determine whether world models stay in the lab or become the default execution layer for reasoning agents.

## What's still open

**How can world models maintain physical and semantic consistency over long-horizon rollouts with temporally abstract actions without suffering from compounding latent-space drift?** Every current system shortens the imagination horizon or relies on heuristic resets to avoid divergence, yet human-level planning requires dreaming across dozens of macro-actions. Research needs a mechanism that constrains the latent transitions to lie on a physically plausible manifold while still allowing the controller to explore new strategies.

**Can a verifier-guided reward signal generalize across radically different environments, or is a separate verifier required per domain?** RLVR-World shows verifiable rewards improve planning, but the verifier is trained on the same distribution as the world model. The open question is whether we can learn one verifier (or a verifier regressor) that scores imagined transitions in unseen domains, effectively bootstrapping world models for transfer without additional real-environment data.

**What abstractions should world models learn when the agent’s sensors are language or tool interfaces instead of pixels?** Reinforcement World Model Learning for LLM-based Agents (2026) reveals the promise of treating latent narratives as dreams, but the field lacks a formal vocabulary for the semantics of those dreams. We need formal objectives that ensure the latent states capture meaning, not just pattern, across modalities.

## Where to read next

If you want to trace how perception and latent compression work, → [Variational Autoencoders](../../02-generative-modeling/concepts/variational-autoencoders.md) grounds the ELBO and latent regularization underpinning the perception module. For the dynamics and planning side, → [[model-based-reinforcement-learning]] shows the general recipe for using learned forward models and contrasts the world-model style with classical MPC. The engineering counterpart to the verifier ideas is → [[latent-dynamics]] where you see how RNNs and diffusion models are currently used to roll out tens of thousands of trajectories cheaply.

## Build it

Training a CartPole world model and solving the environment purely inside its imagination proves that the same architecture that runs Doom dreams for Ha et al. can be compressed to a textbook RL task. The build below shows how to combine a VAE, an RNN transition model, and a controller, and how to verify the dream using real steps.

**What you're building:** A CartPole-v1 latent world model where a VAE compresses the frames, an RNN predicts future latents, and a dream-policy solves CartPole before acting in the real environment.

**Why this is valuable:** The recipe forces you to learn the three-component Ha et al. design, exposes you to the compounding-drift failure mode, and gives you a reproducible artifact (a checkpoint + rollouts) that you can extend to vision-language or tool-using agents.

**Stack:**
- **Model:** mradermacher/WorldModel-Stabletoolbench-Llama3.1-8B-i1-GGUF and mradermacher/WorldModel-Stabletoolbench-Qwen2.5-7B-i1-GGUF — pre-trained inference stacks you reference for verifier architecture and prompt evaluation.
- **Dataset:** Collect `CartPole-v1` trajectories via `gym` (seeded with 123) and store them in a Hugging Face Dataset created with `datasets.Dataset.from_dict`.
- **Framework:** PyTorch 2.1 with `torchvision` 0.15, `gymnasium` 1.1, and `optax`-style optimizers from `torchrl`.
- **Compute:** Free Colab T4 (16 GB VRAM) for ~2 hours of training plus 15 min evaluation.

**The recipe:**
1. Install `pip install torch==2.1 torchaudio torchvision torchrl gymnasium datasets tqdm` and import `gymnasium`, `torch`, `torch.nn`, `TensorDict`, and `datasets`. Set up a `gymnasium.make("CartPole-v1")` environment and collect 5k episodes using a random policy; save frames and actions to disk with `datasets.Dataset`.
2. Preprocess by resizing each frame to 64×64, normalizing pixel values to \([0,1]\), and stacking four consecutive frames as the observation \(x_t\). Use a 4-layer CNN encoder and decoder for the VAE; encode \(x_t\) to \(z_t \in \mathbb{R}^{64}\), and store action \(a_t\in\{0,1\}\) and reward \(r_t\).
3. Train the VAE by minimizing the ELBO with a batch size of 256, learning rate \(1e-3\), and KL weight schedule that ramps from 0 to 1 over 10k steps. Then freeze the encoder and train an LSTM transition \(h_{t+1}=f_\theta(h_t,z_t,a_t)\) and latent predictor \(g_\theta(h_{t+1})\) for 20k steps using teacher-forced latent targets, keeping a small predictive variance mask to emulate stochastic transitions.
4. Train a controller \(\pi_\omega(h_t,z_t)\) using CMA-ES (library `nevergrad` optional) inside a dreamed rollout of 30 steps. Rollout the RNN 30 times per gradient evaluation, compute imagined rewards with a linear head on \(h_t\), and update \(\pi_\omega\) to maximize cumulative imagined reward. Every 500 imagined updates, execute the controller in the real CartPole environment and log the real returns.
5. Evaluate by generating 100 real CartPole episodes with the controller, measuring average reward and the mean KL between imagined and real latent distributions. Save the world model checkpoint and a set of imagined rollouts for later debugging.

**Expected outcome:** A world-model checkpoint that solves CartPole-v1 with >195 reward after ≈300 dreamed training epochs, a set of imagined rollouts, and logs showing that dream reward correlates with real reward most of the time.

- **CS student:** Run the same recipe on an RTX 4070-laptop by cutting the dataset to 2k episodes, reducing the RNN hidden size to 128, and training the controller with PPO for 2 hours; the artifact is the checkpoint with logs recording dream vs. real reward correlation.
- **Applied engineer:** Quantize the frozen world model (VAE + RNN) to INT8 using `torch.quantization.quantize_dynamic`, serve it alongside a FastAPI endpoint, and build a verifier inspired by the Hugging Face world model weights so that a policy request checks dream consistency before executing real actions; autoscale the endpoint on two A10 instances with latency <120 ms.
- **Applied researcher:** Hypothesize that contrastive latent prediction reduces drift: add a contrastive loss between sampled latents and negatively sampled latents from perturbed actions, retrain the transition, and measure whether the verifier loss from RLVR-World-style weighting improves long-horizon imagined reward quality compared to the baseline.
- **Frontier researcher:** Test the open question about abstracted actions by introducing macro-actions (pairs of primitive CartPole moves) into the dream loop, measure drift with the verifier, and report whether simulator rollouts still match imagined statistics when horizon extends to 100 macro steps; this falsifies the claim that short horizons suffice for safety.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*