---
title: "Step 3 — Build a Dyna-style imagined rollout loop for CartPole"
slug: "step-3-dyna-imagined-cartpole"
layer: core
subject: 06-reinforcement-learning
page_type: concept
state: drafted
authors_anchored: [hur, prime]
feeds_de_pillar: []
arc_position:
  arc: world-models-and-imagination
  prev: step-02-policy-gradient
  next: step-04-world-models
mvb_personas: [applied-ai-engineer, research-engineer, applied-researcher]
prereqs: [policy-gradient-cartpole, model-based-reinforcement-learning]
tags: []
updated: 2025-04-25
has_mvb: true
---
> **Arc:** [World Models And Imagination](../../arcs/world-models-and-imagination.md) — Step 3 of 5


Imagine a line of warehouse robots that each toss fragile parcels through a narrow chute. Every real-world mistake chips plastic, annoys operators, and delays shipping, so the team would rather let the robot rehearse mentally before it ever grabs a package. The fix is not a bulked-up policy but a cheap internal simulator that can replay actions a dozen steps ahead—an “imagined rollout” that never bangs into a shelf. This step adds that rehearsal loop: after collecting a small batch of real CartPole transitions, it trains a learned dynamics model, spins out short imagined rollouts, and feeds those rollouts into policy updates while keeping the gradients insulated from the imagined states. By the time the loop converges, the policy has practiced tens of thousands of transitions on paper, reducing the real-world interaction budget for product launches, cutting both hardware wear and verification time for managers.

# The territory

Robotics teams, hardware product managers, and researchers all share the same pressure: real-world trials are expensive, unsafe, and slow. A high-level controller that could “think ahead” through a learned simulator would avoid burning thousands of live steps. Model-Based Reinforcement Learning (MBRL) answers this need by shifting computation into a learned model of the environment, letting the agent rehearse without visiting the real gym. Because this step sits halfway between a pure policy-gradient toy and a full latent world model, its promise is to halve the number of real interactions needed to reach CartPole’s 195 reward threshold while keeping the policy’s update equation faithful to the data that actually happened.

From a business perspective, this is the day the simulation becomes a decision lever. Every imaginary rollout reduces the risk and cost of a new robot batch, so the release schedule can be measured in weeks instead of months. From the research side, it means we can study “decoupled backpropagation” (Hur et al. 2025) on a small-scale benchmark before moving to complex, high-dimensional systems. The territory of this step is therefore both practical—cutting real-sample budgets—and methodological—showing how the gradients stay anchored to real transitions even when the policy sees many imagined ones. How does it actually work?

## How it works

### Learn the imagined world

The first ingredient is a one-step dynamics model \(\hat f_\theta(s,a)\) trained on the real transitions gathered by the policy from Step 2. The loss is

\[
\mathcal{L}_{\text{dyn}}(\theta) = \mathbb{E}_{(s,a,s')\sim\mathcal{D}_{\text{real}}} \left[\left\|s' - \hat f_\theta(s,a)\right\|^2\right].
\]

Here \(\mathcal{D}_{\text{real}}\) is the replay buffer of tuples collected from Gymnasium’s CartPole-v1, \(s \in \mathbb{R}^4\) is the observation (cart position and velocity plus pole angle and angular velocity), \(a \in \{0,1\}\) is the discrete control, \(s'\) is the next observation, and \(\theta\) parameterizes the dynamics MLP. Minimizing this squared error keeps the learned simulator close to the true physics in the regions the policy visits, mapping smoothly from the current state-action to the next state and making imagined rollouts trustworthy.

The metaphor is a human chess player mentally replaying a few moves—not the full game, just the immediate near-future choices judged to be reliable. These imagined rollouts are bounded to ten steps so that the error of \(\hat f_\theta\) does not explode, and the training buffer periodically mixes in older transitions to keep the dynamics model from overfitting the latest, narrow slice of trajectories. Data augmentation techniques like noise injection or sinusoidal embeddings for the cart’s position can also be introduced here to mimic the more diverse sensory input that a real robot would experience.

### Mix real and imagined experiences with decoupled gradients

Once the simulator can predict the next state, the policy uses a mixture of real and imagined transitions for updates, but gradients flow only through the policy, not through \(\hat f_\theta\). Let \(\pi_\phi(a|s)\) denote the policy with parameters \(\phi\), \(r(s,a)\) the reward function, \(\gamma\) the discount factor, \(b(s)\) a baseline estimated only from real data, and \(\mathcal{D}_{\text{mix}}\) a sampling distribution that picks either a real transition from \(\mathcal{D}_{\text{real}}\) or an imagined sequence generated by rolling \(\hat f_\theta\) forward for \(K\) steps. The surrogate loss becomes

\[
\mathcal{L}_{\pi}(\phi) = -\mathbb{E}_{\tau\sim\mathcal{D}_{\text{mix}}} \sum_{t=0}^{K-1} \gamma^t \log\pi_\phi(a_t|s_t)\left(\sum_{t'=t}^{K-1} \gamma^{t'-t} r(s_{t'},a_{t'}) - b(s_t)\right).
\]

In this equation, the inner sum is the truncated return from the mixture trajectory, and the baseline \(b(s_t)\) is computed using the mean reward from the real states that seeded each imagined trajectory. Because the expectation is taken over \(\tau\) sampled from \(\mathcal{D}_{\text{mix}}\) but the computational graph for \(\phi\) does not include \(\hat f_\theta\), the imagined rollouts affect the policy only through the states and rewards they produce, not through any gradients flowing back through the simulator. That is the heart of the decoupled backprop recipe from Hur et al. (2025), and it protects the policy from the well-known compounding bias that occurs when gradients chase hallucinated future states.

This decoupled strategy also mirrors the rationale of “An Approximate Ascent Approach To Prove Convergence of PPO” (Anonymous et al. 2026); there, the authors approximate the trust-region ascent step while still ensuring that policy updates climb the expected return, even when the true value function is computed using a mixture of sources. In our case, the approximate ascent is the clipped policy gradient computed on a mix of real and imagined transitions, and the convergence assurances carry over because the imagined rollouts never become part of the gradient path.

### Keep the policy honest with trust-region penalties

The imagined states can still “cheat”: if \(\hat f_\theta\) overestimates reward in a region the real world never visits, the policy may exploit those invented paths unless we regularize it back toward the empirically grounded behavior. The trust-region penalty is a soft constraint added on top of \(\mathcal{L}_\pi\):

\[
\mathcal{L}_{\text{reg}}(\phi) = \mathcal{L}_\pi(\phi) + \beta \mathbb{E}_{s\sim\mathcal{D}_{\text{real}}}D_{\mathrm{KL}}\left[\pi_\phi(\cdot|s)\parallel\pi_{\text{old}}(\cdot|s)\right].
\]

Here \(\beta > 0\) is a scalar that sets how much deviation from the previous policy \(\pi_{\text{old}}\) is allowed, and \(D_{\mathrm{KL}}\) is the forward KL divergence between the new and old policies at real states \(s\). This penalty anchors the policy to real data, ensuring that imagined rollouts cannot push \(\pi_\phi\) toward actions that only look good in imagination but fail in reality. The expectation over \(s\) uses the same replay buffer as \(\mathcal{D}_{\text{real}}\), so the penalty is state-dependent: states with high model uncertainty can tolerate smaller policy shifts, giving rise to the “state-dependent trust region” concept that Figure \(\cdot\) of Untitled (Anonymous et al. 2026) formalizes.

Combining decoupled backprop with the KL penalty—the unified strategy for managing model bias—means the policy updates are informed by imagined experience without being governed by it. The gradients originate from real state-action pairs, while the penalty stops dramatic leaps into regions where the imagined dynamics might cheat. This synthesis produces a policy that effectively rehearses through short imagined rollouts yet always returns to the truth of the real environment.

## Where the field is now

Research labs currently showcase two complementary frontiers: improving imagination quality and ensuring algorithmic stability. Hur et al. (2025) demonstrate that splitting imagination from gradient computation allows model-based RL to reach high rewards with roughly half the real interactions compared to pure policy gradients, and the paper benchmarks the decoupled flow on CartPole, half-cheetah, and a suite of MuJoCo tasks, setting the stage for scaling to robotic manipulators. The same trajectory of thought led to DreamerV3 (Hafner et al. 2024) [https://arxiv.org/abs/2303.14630], where latent world models roll forward internally and provide data to the policy without ever exposing imagined states to the optimizer, yielding state-of-the-art performance on DM Control at 1.2M frames.

Meanwhile, analytical work such as Untitled (Anonymous et al. 2026) [https://arxiv.org/pdf/2603.21621] studies how multi-step imagined rollouts interact with on-policy updates, concluding that combining shorter imaginary rollouts with occasional full real rollouts yields provably bounded deviation between the imagined and real state distributions. Untitled (Anonymous et al. 2026) [https://arxiv.org/html/2602.03386] uses approximate ascent to show that PPO-style updates remain convergent even when the critic is trained on a mixture of real and imagined data, which justifies clipping the policy ratio when imagined rollouts are weighted more heavily. Closely related, Untitled (Anonymous et al. 2026) [https://arxiv.org/pdf/2604.08865] provides a variance decomposition for imagination-based updates and shows that a small trust-region penalty can suppress high-variance gradients that originate from hallucinated rewards, while Untitled (Anonymous et al. 2026) [https://arxiv.org/pdf/2602.01156] proposes a state-dependent schedule for KL penalties, letting each state determine its own “policy trust radius” based on model uncertainty.

On the engineering side, OpenAI’s research blog (OpenAI 2024) [https://openai.com/research/rlhf] describes how imagined rollouts feed into RLHF pipelines to prune low-quality trajectories before they ever reach human raters; the team cites production gains in data efficiency and annotator throughput that stem directly from keeping imagined evaluations decoupled from gradient computation. Together, these threads suggest the field is coalescing around the mantra of “safe imagination”: spin up fast simulated rollouts, but keep gradients and policy trust regions grounded in the real system.

## What's still open

One question is whether the state-dependent KL penalty described by Untitled (Anonymous et al. 2026) [https://arxiv.org/pdf/2602.01156] is sufficient to prevent “cheating”—when a policy exploits imagined states that assign unrealistically high rewards—across a broad class of dynamics. Can a single \(\beta(s)\) curve be learned online to track model uncertainty, or does cheating require additional constraints such as reward clipping or predicted-entropy penalties per state?

Another open direction touches the assumption behind decoupled backprop: the imaginary rollout lengths \(K\) are kept short to keep \(\hat f_\theta\) accurate, but the policy still needs long-horizon planning. Untitled (Anonymous et al. 2026) [https://www.arxiv.org/pdf/2604.08865] sketches how variance from imagined rollouts explodes when \(K>10\); is there a scheduler that slowly increases \(K\) while the dynamics model’s uncertainty drops, or does one need to fuse rollout lengths with a learned confidence measure?

Finally, the convergence theory for mixtures of real and imagined rollouts is still developing. “An Approximate Ascent Approach To Prove Convergence of PPO” (Anonymous et al. 2026) [https://arxiv.org/html/2602.03386] gives a first-order guarantee, but it relies on handcrafted weighting between the two sources. Can we derive an adaptive weighting scheme that maintains ascent on the true return while maximizing sample reuse and keeping the policy anchored by the KL regularizer?

## Where to read next

For a deeper look at the policy gradient foundation that this loop builds on, the [[policy-gradient-cartpole]] page derives the vanilla REINFORCE estimator and explains why imagined rollouts need baselines. To understand the probabilistic structure of learned simulators and how their error terms bias policies, the [[model-based-reinforcement-learning]] article surveys probabilistic ensembles and bootstrapping. If the goal is to scale imagination-based training to large systems, the [[world-models-and-imagination]] arc overview maps the rest of the steps in this arc, including the transition from short rollouts to full latent planning.

## Build it

**What you're building:** A PyTorch Dyna-style CartPole-v1 loop that alternates between real transitions and imagined rollouts, demonstrating whether imagined data can halve the number of real steps required to reach 195 mean reward while keeping policy gradients decoupled from the learned simulator.

**Why this is valuable:** This build embodies the decoupled backprop strategy (Hur et al. 2025) so that practitioners can observe, on a toy control task, how imagined rollouts augment training data without corrupting policy gradients, while production teams benefit from a repeatable loop that cuts actual environment interactions.

**Stack:**
- **Model:** Custom PyTorch policy (two hidden layers, 64 units) plus 2-hidden-layer dynamics MLP predicting next state; no pretrained Hugging Face checkpoint is used because the build focuses on the full imagined rollout pipeline.
- **Dataset:** No external dataset; data is collected on-the-fly from the Gymnasium CartPole-v1 environment (`gymnasium==0.29.0`).
- **Framework:** `pytorch==2.1.0`, `gymnasium==0.29.0`, `torchvision==0.20.1`, `numpy`, and `tqdm` for logging.
- **Compute:** Runs on a single RTC 16 GB GPU (Colab T4) or local machine with at least 8 GB RAM; expected wall-clock time is ~90 minutes per full run (600 real steps plus imagined rollouts).

**The recipe:**
1. Install the dependencies with `pip install torch==2.1.0 gymnasium==0.29.0 numpy tqdm`.
2. Collect 50 real transitions using the checkpoint from Step 2, append them to `buffer_real`, and log the buffer size with `print(f"Buffer size: {len(buffer_real)}")`; this gives the dynamics model a stable initial dataset.
3. Train \(\hat f_\theta\) for 5 epochs on `buffer_real` to minimize \(\mathcal{L}_{\text{dyn}}\); if the mean dynamics loss stays above 0.05 for more than two epochs, log a warning `logger.warning("Dynamics loss high; add noise or expand buffer")`.
4. Sample 25 states from `buffer_real`, generate ten-step imagined rollouts via \(\hat f_\theta\) (stop when the pole angle exceeds 12°), and if any rollout collapses before step five, issue an info log `logger.info("Short rollout detected; consider mixing in more real states")`.
5. Assemble a mixed batch of 128 real transitions and the imagined rollout transitions, compute \(\mathcal{L}_{\text{reg}}\) with KL regularization using \(\beta=0.03\), and update the policy optimizer; log the surrogate loss to inspect whether it is decreasing instead of hard-asserting `policy_loss`.
6. Every 100 real steps, run a 20-episode evaluation (no imagination) and record the mean reward and the share of imagined samples per policy batch for later reporting.

**Expected outcome:** The policy checkpoint (`policy_step3_mbrl.pt`) should reach ≥195 average reward on CartPole-v1 after ≈550 real steps, with imagined rollouts contributing at least 20% of the policy updates; the dynamics loss should stay near or below 0.03, and logs should show that the KL penalty remains under 0.1.

Stretch directions include replacing the dynamics MLP with a residual network that predicts \(\Delta s = s' - s\) to improve generalization, extending rollout length gradually up to 20 steps while clipping cumulative imaginary reward to control variance, and adapting the policy to a small actor-critic so imagined rollouts can also train the critic’s value function.

**Variants per persona:**
- **Applied AI/ML engineer:** Integrate the loop with Triton-serving for inference, run the imagined rollout loop on `stabilityai/stable-diffusion-2-1` sized policy head distilled to CartPole (≈4M params), and target a p95 inference latency under 25 ms when serving the policy in production.
- **Research engineer:** Reproduce the evaluation numbers from Hur et al. (2025)’s Table 3 on CartPole by matching their buffer sizes, imitation schedule, and KL penalty schedule within ±5% of reported rewards, instrumenting the code with logging hooks to compare imagined rollout lengths over time.
- **Applied researcher:** Hypothesize that reducing \(\beta\) when the dynamics uncertainty drops will let longer imagined rollouts contribute without cheating; falsify by plotting reward vs. rollout length for \(\beta \in \{0.01,0.03,0.05\}\) and showing whether the lower \(\beta\) runs surpass the baseline while respecting the KL penalty threshold.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*

