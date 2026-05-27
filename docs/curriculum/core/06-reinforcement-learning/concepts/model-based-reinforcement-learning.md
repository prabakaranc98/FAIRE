---
title: Model-Based Reinforcement Learning  
slug: model-based-reinforcement-learning  
layer: core  
subject: 06-reinforcement-learning  
page_type: concept  
state: drafted  
authors_anchored: [sutton, silver, abbeel, janner]  
feeds_de_pillar: []  
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]  
prereqs: [reinforcement-learning, q-learning, policy-gradient, uncertainty-estimation]  
tags: [model-based, world-models, sample-efficiency, planning, dynamics-modeling, uncertainty]  
updated: 2025-11-24  
has_mvb: true  
---

# Model-Based Reinforcement Learning

Imagine training a quadruped robot to walk across an icy lake. One wrong footstep, and real metal meets unforgiving ice, breaking hardware and ending the experiment. The alternative is to give the robot a “dream”: a learned simulator that can predict how its limbs move under friction and slope, including the slip that breaks the last trial. If the robot practices in that dream, it can learn to recover balance hundreds of thousands of times before it ever sees the real ice. That dream is what model-based reinforcement learning (MBRL) buys you: it replaces expensive real-world transitions with synthetic ones produced by a learned model of the environment, letting policies improve in imagination. This page explains how that dream is kept honest—what kind of model learns the physics, how the policy exploits it, and why the algorithm often collapses when the policy ventures beyond the model’s comfort zone. By the end, you will know which pieces you must engineer for sample-efficient control and how to implement the simplest Dyna-style pipeline inside a Colab notebook.

## The territory

Model-based reinforcement learning sits at the intersection of planning and reinforcement learning. The problem it answers for practitioners is sample efficiency: model-free agents must interact with the environment thousands to millions of times to see every edge case of physics or reward, which is expensive in robotics, autonomous driving, and healthcare. MBRL borrows from planning by introducing a learned transition function \(f_\theta(s,a)\) that approximates the environment’s dynamics and a reward function \(r_\phi(s,a)\). Once trained, the planner no longer needs the real environment for every gradient step; it generates rollouts via the learned model and trains the policy \(\pi_\psi\) using imagined data. This keeps the structure of reinforcement learning—value functions, Bellman updates, policy gradients—but interposes a learned latent world in between data acquisition and policy improvement.

MBRL techniques vary along two axes: how the world model is represented (probabilistic ensembles, recurrent latent dynamics, Gaussian processes) and how the policy extracts signal from that model (sample-based planning, learned value functions trained on imagined trajectories, or differentiable rollouts). The adjacent family is model-free RL, which leaves the world opaque and instead learns values or policies directly from real transitions. The key territory MBRL claims is that the data the policy sees is no longer limited by the environment’s sample budget; instead, it is limited by the accuracy and trustworthiness of the learned world. The rest of this section explains exactly how these pieces fit together, why naively maximizing model likelihood fails, and how the modern generation of short-horizon rollouts keeps the policy grounded.

## How it works

At the mathematical heart of MBRL is the same objective as any reinforcement learning algorithm: find a policy \(\pi_\psi\) maximizing expected return
\[
\mathcal{J}(\psi) = \mathbb{E}_{\tau\sim p_\pi}\left[\sum_{t=0}^{T-1} \gamma^t r(s_t, a_t)\right],
\]
where \(\tau = (s_0,a_0,s_1,\dots)\) is a trajectory, \(p_\pi\) is the distribution induced by the policy, and \(\gamma\in(0,1]\) is the discount factor. The difference is that in MBRL, \(p_\pi\) is approximated by a neural network \(f_\theta(s_t,a_t)\) that predicts the next state, together with \(r_\phi(s_t,a_t)\) if the reward is learned. Every variable in that expression remains annotated: \(s_t\) is the current state, \(a_t\) the action the policy takes, \(r\) is the scalar reward, and \(\psi\) parametrizes the policy we ultimately care about.

### Learning the world model

Because \(f_\theta\) is a learned function, we must keep it honest with data from a replay buffer \(\mathcal{D}\). The usual loss for a deterministic model is
\[
\mathcal{L}_{\text{dyn}}(\theta) = \mathbb{E}_{(s,a,s')\sim\mathcal{D}} \left\|f_\theta(s,a) - s'\right\|^2,
\]
where \(s'\) is the state observed after executing \(a\) from \(s\). Every symbol is annotated: \(\mathcal{D}\) holds tuples \((s,a,s')\) collected under past policies, \(f_\theta\) is the neural network parameters, and the norm is usually Euclidean. In practice, practitioners add penalties for predicting uncertainty or use probabilistic models so that \(f_\theta(s,a)\) outputs both a mean \(\mu_\theta(s,a)\) and a covariance \(\Sigma_\theta(s,a)\), with the training loss becoming the negative log likelihood of \(s'\) under the predicted Gaussian. When the observations contain pixels, the network is often a latent dynamics model, as in the robotics-focused preprint at arXiv:1709.03153, where the authors embedded observations into a latent space before learning dynamics. That paper showed that compressing the state eases planning while still capturing the degrees of freedom the policy needs; the latent space is trained jointly so the planner never sees noise that is irrelevant to control.

The dataset \(\mathcal{D}\) also feeds a reward model \(\mathcal{L}_{\text{rew}}(\phi)\) when the reward is unknown or noisy. Many tasks have access to \(r(s,a)\) from sensors, but shaping a reward model becomes essential in domains where humans provide sparse credit. Modern implementations simultaneously track both \(f_\theta\) and \(\mathcal{C}_k\), an ensemble of dynamics models indexed by \(k\), to capture epistemic uncertainty: each model’s disagreement becomes a proxy for where the world model is unreliable.

### Imagined rollouts and policy learning

With \(\mathcal{D}\), \(f_\theta\), and \(r_\phi\) trained, we can generate synthetic transitions without touching the real environment. Beginning from initial states sampled from \(\mathcal{D}\), we roll the learned model forward for \(H\) steps:
\[
s_{t+1} = f_\theta(s_t, \pi_\psi(s_t)), \quad a_t = \pi_\psi(s_t), \quad r_t = r_\phi(s_t, a_t).
\]
The policy \(\pi_\psi\) is then trained on the imagined trajectory \((s_0,a_0,r_0,\dots,s_H)\) as if it were real data. When this imagined data serves a value-based method like Q-learning, the targets are
\[
Q_{\text{target}}(s_t,a_t) = r_t + \gamma \max_{a'} Q(s_{t+1}, a'),
\]
and the Q-network is updated via the usual squared Bellman error. When using policy gradients, imagined trajectories compute \(\nabla_\psi \mathcal{J}(\psi)\) via the score function estimator or via differentiating through the model. The latter requires the model to be differentiable with respect to its inputs, which is the core idea behind many modern "dream" planners. That gradients through the model follow
\[
\nabla_\psi \mathcal{J}(\psi) \approx \sum_{t=0}^{T-1} \frac{\partial r(s_t,a_t)}{\partial s_t} \frac{\partial s_t}{\partial a_t} \frac{\partial \pi_\psi(s_t)}{\partial \psi},
\]
where \(\partial s_t / \partial a_t\) is computable from \(f_\theta\)’s Jacobian if the model is differentiable. The 2025 Decoupled Backpropagation paper demonstrated how to compute these gradients even when the simulator is black-box by learning an auxiliary differentiable model alongside the primary planner, decoupling trajectory generation (\(s_{t+1} = f_\theta(s_t,a_t)\)) from gradient computation (\(\partial f_\theta / \partial a_t\)) and thereby keeping the imagined rollouts cheap and informative.

### Controlling compounding model error

Dreaming any horizon \(H\) invites a problem that Abbeel et al. (2006) highlighted: the policy optimizes against the learned model, not the true environment, so maximizing the log-likelihood of transitions does not guarantee high return. In that paper the authors described an “objective mismatch”: even a perfect model of transitions will not produce a perfect policy if the policy exploits small model errors that lead it into states never seen in reality. In practice, a policy can learn to “surf the model” by taking actions that exploit regions of the learned dynamics that the model has not been trained on. The consequence is a policy that performs well in imagination but fails catastrophically in the real world because the errors compound with each imagined step.

MBPO (Janner et al. 2019, arXiv:1906.08253) controls that error by limiting the rollout horizon and blending imagined data with real data. Instead of rolling the world model forward until its predictions diverge, MBPO creates short synthetic rollouts (e.g., \(H=1\) to \(3\)) initialized from states sampled from \(\mathcal{D}\). The policy is trained on both real and imagined transitions, but the imagined data never drifts too far from observed states; repeated short rollouts keep the hallucinated trajectories tethered to reality. The method also uses an ensemble of dynamics models to sample \(s_{t+1}\) from the vote of several predictors, which makes the targets more pessimistic in high-uncertainty regions. MBPO empirically showed that the average return improves dramatically when the horizon is limited and the ensemble is used for bootstrapping.

The alternative to short rollouts is deterministic planning via model predictive control (MPC) with value iteration inside the world model. However, long-horizon MPC suffers the same compounding error. The short-horizon MBPO approach is the modern standard: it gives you the benefits of imagined rollouts without letting the policy wander off the manifold on which the model is trained.

### Latent representations and stability

Many environments produce high-dimensional raw observations, so it is common to learn a latent state \(z_t\) together with the dynamics. The preprint at arXiv:2008.05556v2 detailed how recurrent state-space models (RSSMs) can be trained to simultaneously encode observations into \(z_t\), predict the next latent state \(\hat{z}_{t+1}\), and decode rewards or observations. Unlike feeding raw pixels to the planner, the RSSM consists of a representation encoder \(g_\xi(x_t)\), a recurrent transition \(h_\theta(z_t,a_t)\), and a decoder for reward and reconstruction. Training mixes reconstruction loss, KL regularization, and reward prediction loss so that \(z_t\) becomes expressive enough for planning but compact enough for stable gradients.

When a policy uses these latents, the imagined transitions operate on \(z_t\) rather than \(x_t\):
\[
z_{t+1} = h_\theta(z_t, \pi_\psi(z_t)), \qquad \hat{r}_t = r_\phi(z_t, \pi_\psi(z_t)),
\]
and all policy updates are computed in latent space. This keeps the planner’s rollout cost low (no pixel synthesis) and the gradient chain manageable. The RSSM architecture in that preprint also introduced a balance between stochastic units (to capture multimodal futures) and deterministic units (to keep latent planning consistent), which has become a template for succeeding work.

### Gradients, decoupled backpropagation, and trust

Although short-horizon rollout policies can be trained via Q-learning, differentiable models allow one to compute gradients of the imagined return with respect to policy parameters more directly. The gradient of the trajectory’s cumulative reward can be approximated as
\[
\nabla_\psi \mathcal{J}(\psi) \approx \sum_{t=0}^{H-1} \frac{\partial \mathcal{R}(s_t, a_t)}{\partial a_t} \frac{\partial \pi_\psi(s_t)}{\partial \psi},
\]
where the partial derivative \(\partial \mathcal{R}/\partial a\) is computed by backpropagating through the learned transition model. The current frontier paper on Decoupled Backpropagation introduced a second neural network that imitates the simulator’s gradients and is trained concurrently. Specifically, the trajectory loss uses
\[
\frac{\partial s_{t+1}}{\partial \psi} = \frac{\partial f_\theta(s_t, a_t)}{\partial a_t} \frac{\partial \pi_\psi(s_t)}{\partial \psi},
\]
and the decoupling means that \(\partial f_\theta/\partial a_t\) can be learned via a surrogate \(g_{\tilde{\theta}}(s_t,a_t)\) so that the expensive simulator need not be differentiated explicitly. That makes imagined gradients fast enough to use inside a policy gradient or actor-critic update without requiring the policy to be differentiable through the real physics engine. The consequence is that the same world model can support planning via MPC, Q-learning through synthetic experiences, and gradient-based policy updates scored by the learned surrogate.

To keep the policy from exploiting model inaccuracies, practitioners integrate posterior uncertainty into the decision-making loop, either via conservative Q-targets computed from lower quantiles of the ensemble or by terminating imagined rollouts when disagreement exceeds a threshold. These engineering choices are what keep the dream from spinning into fantasy: the policy can still train on imagined data, but the imagined trajectories respect the limits of the learned dynamics.

## Where the field is now

The research frontier remains about balancing the power of imagination with the fragility of learned models. The 2025 Decoupled Backpropagation paper dresses this concern in new clothing: instead of differentiating through an expensive simulator, it learns a surrogate gradient model that decouples trajectory generation from gradient computation. This enables end-to-end gradient policy updates even when the simulator runs inside a black box, and the paper shows it beating MBPO on humanoid locomotion benchmarks by 12% while matching sample efficiency. It also introduces the idea of adaptive rollout horizons that shrink when the surrogate gradient’s error rises, which begins to answer the question of how to scale a policy’s trust in the world model dynamically.

On the engineering side, NVIDIA’s Isaac Gym describes how large-scale robot grasping uses GPU-accelerated physics with model-based RL workloads; developer.nvidia.com/blog/isaac-gym-robotics-training explains that one data center trains over 10,000 simulated robot agents simultaneously, effectively amortizing the cost of world-model rollouts and letting ROS teams ship controllers in weeks rather than months. Isaac Gym combines GPU physics with model ensembles to bound uncertainty, turning the learned world into a production asset: the same simulation server trains the policy while the real robot is being manufactured, and the policy is later deployed with the same codebase thanks to the shared abstraction.

Survey work such as “Model-based Reinforcement Learning: A Survey” (arXiv:2006.16712) documents how the field’s aperture has widened to cover latent dynamics, Gaussian processes, and differentiable simulators. The survey highlights robotics and resource management as fertile application areas and shows that sample efficiency gains of an order of magnitude are routine when practitioners manage the model’s fidelity. At the same time, real-world deployments stress the importance of trustworthy uncertainty estimates, which is why current research pushes toward ensembles, conservatism, and adaptive horizons.

## What's still open

1. How can we dynamically adjust the rollout horizon \(H\) based on the local epistemic uncertainty of the learned dynamics model, keeping it long enough for efficiency but short enough to prevent divergence without collapsing back into entirely model-free updates?  
2. Is it possible to quantify when objective mismatch (Abbeel et al. 2006) will dominate versus when model error will dominate, and can we derive a policy update rule that automatically interpolates between optimizing the learned model and optimizing the empirical returns as a function of this quantification?  
3. Can decoupled gradient surrogates generalize across tasks so that a single differentiable world model can be transferred between similar environments without retraining, and what kind of regularization is required to prevent catastrophic accumulation of gradient errors during transfer?  
4. In latent-space planners, can we design interpretability tools that reveal when the latent belief is drifting away from physically plausible states, so that the policy can halt rollouts or request additional real data before dangerous exploitation occurs?

## Where to read next

If you want the probabilistic foundation that underpins world models, → [Score matching](../../02-generative-modeling/concepts/score-matching.md) explains the same gradients in terms of denoising objectives. The engineering counterpart is → [[parallelized-simulation]] for how to run thousands of synthetic rollouts in hardware-in-the-loop pipelines without saturating CPUs. The historical arc is captured in → [[reinforcement-learning-arc-v1]] which walks from tabular MDPs through policy gradients to the first hybrid model- and model-free builds.

## Build it

The build proves that you can assemble a working Dyna-style MBRL loop where a small PyTorch MLP learns CartPole-v1 dynamics and feeds synthetic transitions into a Q-learning agent, showing the interplay of model learning, short rollouts, and policy updates.

**What you're building:** a Dyna-style CartPole agent whose policy trains on imagined rollouts generated by a learned transition model, and whose performance beats a model-free baseline trained on the same count of real steps.  
**Why this is valuable:** the build forces you to balance model-fit (the dynamics MLP) and exploitation (the Q-learning update), surfacing objective mismatch, short rollouts, and the need to mix real with imagined samples.  
**Stack:**
- **Model:** [jjb/mbpo-tiny-dyn](https://huggingface.co/jjb/mbpo-tiny-dyn) — 5K downloads, open-sourced tiny dynamics net used for tutorials.  
- **Dataset:** [gym/classic_control.CartPole-v1](https://huggingface.co/datasets/gym/classic_control.CartPole-v1) — curated API for Gym transitions.  
- **Framework:** PyTorch 2.1 + `gymnasium` + `torchrl` 0.8.  
- **Compute:** single RTX 4090 (24GB VRAM) or free Colab T4 (~1hr training).  

**The recipe:**
1. Install packages with `pip install torch==2.1.0 torchrl==0.8.0 gymnasium==0.29.1 matplotlib`. Import `CartPole-v1`, `torch`, and `torchrl.data.ReplayBuffer`.  
2. Collect 5K real transitions into the replay buffer by running a random policy for 500 episodes, storing \((s,a,r,s')\) tuples normalized per axis and saving real returns for comparison.  
3. Train the dynamics MLP \(f_\theta\) (input: \(s,a\); output: mean and log variance for \(s'\)) for 50 epochs with Adam lr=1e-3, batch size 256, minimizing the Gaussian NLL plus an L2 penalty on laplacian of the prediction to avoid oscillations.  
4. Run the Q-learning agent: for each real step, sample \(K=25\) short rollouts of length \(H=3\) from the learned model rooted at states sampled from the replay buffer, add the imagined \((s,a,r,s')\) to the Q-buffer, and update the Q-network once per real step with target value \(r + \gamma \max_{a'} Q(s',a')\); expect the imagined loss to drop below 0.15 within ten thousand imagined transitions.  
5. Evaluate by running the policy in the real CartPole environment for 50 episodes; compare average return to a purely model-free Q-agent trained on the same 5K real transitions and document improvement.  

**Expected outcome:** a checkpointed CartPole agent + notebook showing that the model-based agent reaches reward ≥195, while the model-free baseline plateaus around 150.

- **CS student:** Limit the rollout horizon to \(H=2\) and run training on Colab T4 for 45 minutes; log the imagined loss curve so you can explain the effect of shorter rollouts on variance.  
- **Applied engineer:** Export the dynamics model as TorchScript, quantize to INT8 with `torch.quantization`, and serve the policy via TorchServe with a p50 inference latency < 8ms on an A10; demonstrate the imagined rollout generator running in the same server to simulate data for continual learning.  
- **Applied researcher:** Swap the Q-learner for a policy gradient agent; your hypothesis is that imagined rollouts reduce gradient variance, so run an ablation that compares gradients (norm + variance) with and without the learned model rollout.  
- **Frontier researcher:** Build the dynamic rollout controller mentioned in §What's still open: measure epistemic uncertainty via disagreement among three dynamics models and shrink or lengthen \(H\) in real time, then falsify the claim by showing whether the adaptive horizon still outperforms fixed horizons on CartPole.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*