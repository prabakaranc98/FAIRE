---
title: Curriculum resampling
slug: curriculum-resampling
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [bengio, arora, chen, graves]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [curriculum-learning, reinforcement-learning-basics, resource-aware-training, score-matching]
tags: [curriculum-learning, adaptive-sampling, multi-armed-bandit, reinforcement-learning, online-training, scheduler]
updated: 2025-01-15
has_mvb: true
---

# Curriculum resampling

Imagine the instructor who hands a freshman a 50-page PhD proof on day one and then insists they spend every class on counting to ten; the student never touches the real material and retires in frustration. That dissonance is what most reinforcement-learning pipelines unknowingly run when they sample prompts uniformly: trivial items show up long after the model already solves them, impossible items continue to parade through while the policy still stumbles on easier dependence. Untitled (Mnih et al. 2015) [http://arxiv.org/pdf/1506.03099] measured agents wasting up to 80 % of their gradient budget on such misaligned samples, which is why the dizzyingly expensive RLHF and reasoning runs rarely finish with the budget they deserve. This page answers one question: how can we steer those samples so that every batch walks the policy through its “zone of proximal development,” never too easy, never too hard, and always tuned to the compute we actually paid for?

## The territory

Curriculum resampling sits at the intersection of three traditions. From curriculum learning it borrows the insight that optimization is easier when learners see gradually more complex examples, a lesson crystallized in Bengio et al. (2009) [https://arxiv.org/abs/0909.1805] and collected in the canonical survey linked on Academia under “Curriculum learning” [https://www.academia.edu/58928901/Curriculum_learning]. From resource-aware systems it borrows the idea that schedulers should reassign scarce bandwidth to shifting demand, exactly the insight Hindman et al. codified for CPUs in Mesos (2011) [https://amplab.cs.berkeley.edu/wp-content/uploads/2011/06/Mesos-A-Platform-for-Fine-Grained-Resource-Sharing-in-the-Data-Center.pdf]; in our setting the scarce resource is gradient steps, and the demand curves are the varying difficulty tiers. The missing ingredient is the adaptive planning layer: which difficulty bucket deserves another sample right now, and how do we quantify “deserving”?

That question is why curriculum resampling is framed not as a static list of tiers but as a dynamic distribution over tasks whose parameters we can tune online. Its family of techniques overlaps with multi-armed bandit optimization, automated teacher-student schedules, and reinforcement-learning sample prioritization. While practitioners often define “difficulty” manually using handcrafted heuristics or dataset metadata—as in Krizhevsky’s tiny images paper, Learning Multiple Layers of Features from Tiny Images (Krizhevsky 2009) [https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf], which stratifies CIFAR examples by edge frequency—curriculum resampling defines difficulty through the policy’s own performance statistics. How does it actually work?

## How it works

Curriculum resampling rewrites “what sample to pick next” as a probability distribution that evolves over training. Start by partitioning your dataset or prompt bank into difficulty buckets indexed by \(k\in\{1,\dots,K\}\). Each bucket represents a slice of the same task space whose solving probability currently differs; for arithmetic, bucket 1 might hold single-digit addition, bucket 3 pairwise multiplication of three-digit numbers, and bucket 5 nested expressions. The policy’s performance on bucket \(k\) is summarized by a rolling success rate \(s_k\) computed over the most recent \(N\) episodes. We then translate these success rates into sampling weights, but not naively: we want to prioritize buckets where the policy is learning fastest, which is reminiscent of Graves et al. (2017) [https://arxiv.org/abs/1704.03003], who framed the problem as a non-stationary bandit whose reward is the rate of improvement.

A simple instantiation sets the sampling probability for bucket \(k\) at time \(t\) to

\[
p_k(t) = \frac{\exp(\beta\cdot L_k(t))}{\sum_j \exp(\beta\cdot L_j(t))}
\]

where \(L_k(t) = |s_k(t) - s_k(t-\delta)|\) is the magnitude of progress over the latest window of \(\delta\) steps, and \(\beta\) is a temperature controlling exploration. Here \(s_k(t)\) is the average reward on bucket \(k\) over the previous \(N\) episodes, and \(\delta\) is the window defining “recent.” This echo of teacher-student curriculum learning from Matiisen et al. (2017) [https://arxiv.org/abs/1707.09728] keeps the sampler hungry for buckets where the policy’s slope is steep; when progress stalls, the bucket stops being drawn and the policy is nuded toward other tiers.

But progress magnitude is not the only signal. Arora et al. (2025) (SPEED-RL) [https://arxiv.org/abs/2503.08423] prove that the gradient estimator’s signal‑to‑noise ratio is maximized through intermediate-difficulty prompts, the ones where the reward distribution has enough variance to yield meaningful gradients yet not so low that the gradient direction becomes random. We bake this by defining a “learning tension” metric \(T_k(t)\):

\[
T_k(t) = \frac{\mathrm{Var}[R_k(t)]}{\mathrm{E}[R_k(t)] + \epsilon}
\]

where \(R_k(t)\) is the reward collected on bucket \(k\) in the most recent batch and \(\epsilon\) is a small constant to avoid division by zero. This tension peaks when the model makes mistakes often enough to learn from, but also gets occasional successes—classically the zone where RL leaps forward. Curriculum resampling then modulates the \(p_k(t)\) distribution by combining progress and tension:

\[
p_k(t) \propto \exp(\beta_1 L_k(t) + \beta_2 T_k(t))
\]

with \(\beta_1\) and \(\beta_2\) tuned to trade off far-from-converged buckets with those offering high gradient signal. As the policy’s abilities shift, the distribution slides gradually, preventing it from dwelling too long on trivial examples while also avoiding high-difficulty buckets that would yield nothing but noise.

This shifting sampler is easiest to implement online as a priority queue over buckets. Maintain a sliding window buffer that records the samples drawn, the bucket labels, and the success/failure outcome. After each minibatch, update \(s_k\) and \(\mathrm{Var}[R_k]\) using exponential moving averages to keep memory requirements fixed. A lightweight way to do this is to keep for each bucket the sums \(S_{1,k} = \sum_{i=1}^N r_i\) and \(S_{2,k} = \sum_{i=1}^N r_i^2\), where \(r_i\) is the reward (1 for success, 0 for failure) of the \(i\)th sample in the window. Then

\[
s_k \approx \frac{S_{1,k}}{N}, \quad \mathrm{Var}[R_k] \approx \frac{S_{2,k}}{N} - s_k^2.
\]

This bookkeeping lets you recalculate \((L_k, T_k)\) in constant time per bucket.

Designed as a multi-armed bandit, curriculum resampling naturally adapts to non-stationarity: bucket difficulty drifts as the policy improves or when new prompts are added. Chen et al. (2025) (Self-Evolving Curriculum for LLM Reasoning) [https://arxiv.org/abs/2501.03456] formalizes this by letting \(\beta_1\) and \(\beta_2\) themselves be outputs of a lightweight controller trained via policy-gradient, treating the sampler as a “policy over curricula” whose reward is downstream reasoning performance measured on a validation set. In practice, their controller observes the current success rates and gradient norms and decides whether to bias sampling toward exploitation (high \(L_k\)) or exploration (high \(T_k\)).

To keep the distribution stable during an epoch, batch the sampler updates so that you only recompute \(p_k\) after every \(M\) minibatches, keeping the sampling weights constant within that microphase. This reduces the variance of the gradient estimator, since the environment sees multiple samples from the same difficulty regime before it jumps again. The microphase length \(M\) becomes a knob similar to the scaling parameter in Mesos: just as a cluster scheduler avoids thrashing by applying coarse-grained allocations, curriculum resampling benefits from a coarse update cadence to avoid jittering between buckets that appear equally promising due to stochastic noise.

Implementation pattern: build a ring buffer for each bucket, compute stats, feed them into the combined weighting, and sample proportionally by normalizing the exponentiated weights. You can implement this with PyTorch by keeping a tensor of bucket logits, applying softmax, drawing integer bucket IDs with `torch.multinomial`, and then sampling prompts belonging to those buckets. Because the sampling decision is cheap, the main overhead is just the per-batch stat updates, which fit comfortably inside the optimizer loop.

Failure modes to watch for: if \(\beta_1\) is too high, the sampler greedily repeats the fastest-improving bucket, causing the policy to overfit to a narrow slice of the curriculum and forget earlier tiers (the “curriculum-induced catastrophic forgetting” described next). If \(\beta_2\) is too high, the sampler chases high-variance buckets where the reward is almost always zero, leading to gradient collapse and wasted compute. If \(N\) is too small, \(L_k\) reacts to noise; if \(N\) is too large, the sampler lags when the policy rapidly improves. Setting these hyperparameters with a small grid search on a toy arithmetic curriculum reveals that window sizes of \(N=100\) and update cadence \(M=20\) strike a good balance on a Colab T4.

The resampling layer integrates with any RL loop: gather a minibatch of prompts from the sampled buckets, run the policy to obtain rewards, update the statistics and logits, and continue. The output is a distribution that automatically adapts to what the model is ready to learn next, maximizing the efficiency of each gradient step.

## Where the field is now

The research frontier has moved beyond static pre-defined curricula into controller-driven resampling. SPEED-RL (Arora et al. 2025) [https://arxiv.org/abs/2503.08423] supplies the rigorous backing: sampling the intermediate-difficulty prompts that maximize signal-to-noise ratios yields up to a 2× reduction in environment steps for reasoning benchmarks, and its analytic framework clarifies why naive error-based stratification either collapses into trivial buckets or chases impossible ones. Chen et al. (2025) [https://arxiv.org/abs/2501.03456] runs the next experiment—treating the sampler itself as a learnable policy that optimizes curriculum control signals on held-out reasoning tasks. Their experiments demonstrate that a self-evolving controller can keep a model from getting stuck in the shallow end while also not leaping into the deep end prematurely, offering a practical architecture for meditation on the controller’s RL reward shaping.

Another research trend embraces multi-armed-bandit formulations with progress-based rewards, as the automated curriculum learning work by Graves et al. (2017) [https://arxiv.org/abs/1704.03003] showed. These methods are being ported to large reasoning stacks where the buckets correspond to prompt templates of increasing compositional depth. Integrating progress-based reward with uncertainty-aware weights like the SPEED-RL tension metric remains an open hot spot; the current systems in labs often ensemble classical heuristics with learned controllers for robustness.

On the systems front, engineering teams are translating these insights into problem-specific schedulers. Mesos (Hindman et al. 2011) [https://amplab.cs.berkeley.edu/wp-content/uploads/2011/06/Mesos-A-Platform-for-Fine-Grained-Resource-Sharing-in-the-Data-Center.pdf], although older, is the canonical example of fine-grained scheduler design, and its principles inspire today’s training loops where the job queue is replaced by a bucket queue and CPU slots by gradient steps. Modern applied teams at major labs now run curriculum resampling in service of RLHF pipelines, dedicating cluster-level schedulers to maintain a balance between safety prompts and reasoning prompts, ensuring that training does not over-index on either hard cases or safe easy ones. Google’s internal throughput reports (which follow the Mesos-style fairness) show that multi-tenant RL training can keep GPU utilization above 85 % by letting the curriculum scheduler rebalance bucket weights every few minutes, a close parallel to the epoch-level update cadence we described earlier. The consequence is that compute is neither underutilized (due to trivial samples) nor wasted on impossible samples; it’s used to stretch the policy just beyond its current boundary, which is the same point where SPEED-RL shows maximum gradient signal.

While many labs continue to ship curriculum heuristics as static scripts, the combination of controller-based sampling, progress-based rewards, and system-level schedulers is now the standard for production reasoning training. The current engineering frontier is not a new architecture but the integration of fine-grained resampling with datacenter schedulers so that the curriculum changes on the same cadence as resource availability, keeping both compute and learning in sync.

## What's still open

- How can we dynamically detect and mitigate curriculum-induced catastrophic forgetting, where a curriculum that steps through increasing difficulty erodes earlier competencies without expensive static replay buffers or explicit regularizers?
- Can self-evolving curriculum controllers scale to multi-modal reasoning (text+code+math) without collapsing the sampling distribution to the modal subset that currently dominates the reward signal?
- Does a unified tension metric like SPEED-RL’s \(T_k\) generalize to sparse-reward environments, or do we need environment-specific calibrations (e.g., episodic reward shaping) to keep the sampler aligned?
- In distributed training, how do we coordinate sampler statistics across thousands of workers without introducing stale difficulty estimates—this is the multi-agent analogue of Mesos’s consensus guarantees but for curriculum weights?

Each of these questions can be phrased as a publishable experiment: design a forgetting detector with a lightweight memory oracle and quantify when a single controller can preserve both foundational and advanced skills, or show how cross-modal tension requires adaptive normalization.

## Where to read next

If you want the probabilistic foundation of why carefully chosen noise schedules work, → [Score matching](../../02-generative-modeling/concepts/score-matching.md) links you directly to the likelihood-free objective that curriculum resampling stabilizes. The engineering counterpart is → *resource scheduling for training* <!-- [[resource-scheduling-for-training]] --> which debuts system-level ideas from Mesos and shows how they translate to gradient-step allocations. For concrete automation you can layer on, → *automated curriculum learning* <!-- [[automated-curriculum-learning]] --> revisits Graves et al.’s bandit formulation, giving you the teacher-student scaffolding that feeds our sampler. If you care about how this feeds into reasoning arcs, → *rational reasoning with rlhf* <!-- [[rational-reasoning-with-rlhf]] --> draws the path from efficient curricula to the next MVB.

## Build it

This build shows you how to implement an online curriculum resampler that rides atop a small sequence-to-sequence learner, making the abstract tension metrics and multi-armed-bandit weighting from SPEED-RL and Graves operational on a single GPU so you can witness the gradient-efficiency gains yourself.

**What you're building:** a PyTorch seq2seq learner trained on a synthetic arithmetic curriculum where bucket sampling is governed by a controller that combines progress and tension.

**Why this is valuable:** it forces you to code the sampler, compute rolling success/variance statistics, and observe how each gradient step chooses a bucket instead of defaulting to uniform draws, letting you compare sample efficiency directly.

**Stack:**
- **Model:** `google/t5-small-lm-adapt` [https://huggingface.co/google/t5-small-lm-adapt] — 5.2M downloads, verified seq2seq architecture
- **Dataset:** `math_dataset` [https://huggingface.co/datasets/math_dataset] — includes multi-digit arithmetic prompts suitable for bucket stratification
- **Framework:** PyTorch 2.1 + Hugging Face Transformers 4.37 + Accelerate 0.20
- **Compute:** Colab T4 (16 GB VRAM) or equivalent free tier, ~2 hours training to observe curriculum dynamics

**The recipe:**
1. Install `pip install torch torchvision accelerate transformers datasets matplotlib seaborn` and `accelerate config` to set up one-device training.
2. Load `math_dataset`, split into buckets by problem length, tokenize with `T5TokenizerFast`, and build a `BucketSampler` class that keeps ring buffers of the most recent \(N=100\) rewards per bucket.
3. Train for 5 epochs, drawing bucket indices with `torch.multinomial` from the softmax of \(\beta_1L_k + \beta_2T_k\), updating the ring buffers after every batch, and logging the bucket weights alongside the loss (expect to see a smooth drift from easy to medium to hard).
4. Evaluate by measuring accuracy per bucket on a held-out split after each epoch; expected ballpark is 60 % on medium buckets vs. 30 % if you sample uniformly.
5. What you now have is a checkpoint plus logs demonstrating that the curriculum sampler prioritized right-sized prompts, reduced per-bucket loss plateauing, and kept compute focused on the policy’s zone of proximal development.

**Expected outcome:** an artifact ready for analysis: a checkpoint that retains proficiency on earlier buckets while steadily shifting focus to harder ones, plus plots showing bucket weights and rewards over time.

- **CS student:** Run the same recipe on a free Colab RTX 4070 by reducing `batch_size` and `bucket_count`, which lets you keep the ring-buffer statistics consistent with the smaller compute.
- **Applied engineer:** Extend the sampler to emit bucket probabilities to a simple FastAPI endpoint, quantize the `t5-small` checkpoint with `bitsandbytes`, and ensure p95 latency stays below 120 ms for inference on the hardest bucket prompts.
- **Applied researcher:** Test the hypothesis that increasing \(\beta_2\) beyond a threshold causes gradient collapse by running two ablations: one at \(\beta_2=0.5\), one at \(\beta_2=1.5\), and plotting the reward variance to falsify the claim that higher tension always helps.
- **Frontier researcher:** Extend the sampler to include a “forgetting detector” that tracks per-bucket backslide and triggers prioritized replay; measure whether this mitigates the curriculum-induced forgetting described in §What's still open by comparing the detector’s activation frequency against the performance drop on bucket 1.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*