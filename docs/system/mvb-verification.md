---
title: MVB feasibility + URL trust verification
description: Post-sweep audit. MVB stacks checked against HuggingFace reality + GPU-VRAM math; external URLs scored against the multi-signal trust function.
---

# MVB feasibility + URL trust verification

**Pages scanned:** 25
**MVB stacks checked:** 9 (passed: **5**, failed: **4**)
**External URLs checked:** 221 (trusted: **155**, flagged: **66**)

## MVB feasibility failures

Each line: `{page} :: {persona} → {issue}`. Most common failure modes are phantom HF IDs (model not on the Hub) and incoherent compute-vs-model-size triples.

- docs/curriculum/14-biology-life-sciences/epigenomics.md :: curious learner → Model `co/datasets` does not exist on HuggingFace (not found or private (401)).
- docs/curriculum/08-causal-statistical-inference/do-calculus.md :: cs student / tinkerer → Model `causal-datasets/lalonde` does not exist on HuggingFace (not found or private (401)).
- docs/curriculum/08-causal-statistical-inference/estimation.md :: cs student / tinkerer → Model `facebook/digit-pose-estimation` does not exist on HuggingFace (not found or private (401)).
- docs/curriculum/15-ml-theory-foundations/entropy.md :: cs student / tinkerer → Model `jasonrqh/Math-CoT-44k-Qwen3-32b-n32-16384-with-logprob-and-entropy` does not exist on HuggingFace (not found or private (401)).

## URL trust failures

Each line: `{page} → {verdict}: {url} ({reason})`. NEGATIVE = on the terminal block-list (Medium/Substack/Wikipedia/social media). low-trust = scored below 0.5 on the multi-signal verifier.

- docs/curriculum/07-attention-memory-reasoning/efficient-attention.md → low-trust (0.00): https://github.com/Dao-AILab/flash-attention  (low trust (0.00): no positive signals matched)
- docs/curriculum/07-attention-memory-reasoning/efficient-attention.md → low-trust (0.00): https://github.com/vllm-project/vllm  (low trust (0.00): no positive signals matched)
- docs/curriculum/07-attention-memory-reasoning/efficient-attention.md → low-trust (0.00): https://github.com/vllm-project/vllm  (low trust (0.00): no positive signals matched)
- docs/curriculum/07-attention-memory-reasoning/single-head-attention.md → low-trust (0.40): https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html  (low trust (0.40): +0.40 approved-research-domain)
- docs/curriculum/05-statistical-probabilistic-ml/bayesian-nn.md → low-trust (0.40): https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/  (low trust (0.40): +0.40 approved-research-domain)
- docs/curriculum/05-statistical-probabilistic-ml/em.md → low-trust (0.00): https://github.com/prabakaranc98/FAIRE  (low trust (0.00): no positive signals matched)
- docs/curriculum/05-statistical-probabilistic-ml/em.md → low-trust (0.40): https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html  (low trust (0.40): +0.40 approved-research-domain)
- docs/curriculum/05-statistical-probabilistic-ml/em.md → low-trust (0.40): https://pyro.ai/  (low trust (0.40): +0.40 approved-research-domain)
- docs/curriculum/05-statistical-probabilistic-ml/em.md → low-trust (0.00): https://doi.org/10.1111/j.2517-6161.1977.tb01600.x  (low trust (0.00): no positive signals matched)
- docs/curriculum/05-statistical-probabilistic-ml/em.md → low-trust (0.40): https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/  (low trust (0.40): +0.40 approved-research-domain)
- docs/curriculum/04-neural-networks-dl/backpropagation.md → low-trust (0.00): https://github.com/prabakaranc98/FAIRE  (low trust (0.00): no positive signals matched)
- docs/curriculum/04-neural-networks-dl/backpropagation.md → low-trust (0.40): https://pytorch.org/docs/stable/autograd.html  (low trust (0.40): +0.40 approved-research-domain)
- docs/curriculum/04-neural-networks-dl/backpropagation.md → low-trust (0.40): https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html  (low trust (0.40): +0.40 approved-research-domain)
- docs/curriculum/04-neural-networks-dl/cnn.md → low-trust (0.00): https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862ec079d7d3f139615696cc3173-Paper.pdf  (low trust (0.00): no positive signals matched)
- docs/curriculum/04-neural-networks-dl/cnn.md → low-trust (0.00): https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862ec079d7d3f139615696cc3173-Paper.pdf  (low trust (0.00): no positive signals matched)
- docs/curriculum/04-neural-networks-dl/cnn.md → low-trust (0.40): https://pytorch.org/vision/stable/models.html  (low trust (0.40): +0.40 approved-research-domain)
- docs/curriculum/04-neural-networks-dl/cnn.md → low-trust (0.00): https://github.com/prabakaranc98/FAIRE  (low trust (0.00): no positive signals matched)
- docs/curriculum/04-neural-networks-dl/cnn.md → low-trust (0.40): https://pytorch.org/vision/stable/models.html  (low trust (0.40): +0.40 approved-research-domain)
- docs/curriculum/14-biology-life-sciences/cell-simulation.md → low-trust (0.40): https://arxiv.org/html/2603.25240  (low trust (0.40): +0.40 approved-research-domain)
- docs/curriculum/14-biology-life-sciences/cell-simulation.md → low-trust (0.40): https://arxiv.org/html/2604.27646v1  (low trust (0.40): +0.40 approved-research-domain)
- docs/curriculum/14-biology-life-sciences/cell-simulation.md → low-trust (0.40): https://arxiv.org/html/2603.25240  (low trust (0.40): +0.40 approved-research-domain)
- docs/curriculum/14-biology-life-sciences/cell-simulation.md → low-trust (0.40): https://ar5iv.labs.arxiv.org/html/2210.14330  (low trust (0.40): +0.40 approved-research-domain)
- docs/curriculum/14-biology-life-sciences/cell-simulation.md → low-trust (0.40): https://arxiv.org/html/2408.12373  (low trust (0.40): +0.40 approved-research-domain)
- docs/curriculum/14-biology-life-sciences/cell-simulation.md → low-trust (0.00): https://www.gene.com/scientists/publications  (low trust (0.00): no positive signals matched)
- docs/curriculum/14-biology-life-sciences/cell-simulation.md → low-trust (0.00): https://insilico.com/research  (low trust (0.00): no positive signals matched)
- docs/curriculum/14-biology-life-sciences/cell-simulation.md → low-trust (0.00): https://github.com/scverse/scvi-tools  (low trust (0.00): no positive signals matched)
- docs/curriculum/14-biology-life-sciences/cell-simulation.md → low-trust (0.00): https://github.com/li-lab/cellflow  (low trust (0.00): no positive signals matched)
- docs/curriculum/14-biology-life-sciences/epigenomics.md → low-trust (0.00): https://www.illumina.com  (low trust (0.00): no positive signals matched)
- docs/curriculum/14-biology-life-sciences/epigenomics.md → low-trust (0.00): https://www.pacb.com  (low trust (0.00): no positive signals matched)
- docs/curriculum/14-biology-life-sciences/epigenomics.md → low-trust (0.00): https://hdl.handle.net/2445/225034  (low trust (0.00): no positive signals matched)
- docs/curriculum/12-physics-scientific-ai/equivariant-networks.md → low-trust (0.00): https://developer.nvidia.com/blog/accelerate-drug-and-material-discovery-with-new-math-library-nvidia-cuequivariance/  (low trust (0.00): no positive signals matched)
- docs/curriculum/12-physics-scientific-ai/equivariant-networks.md → low-trust (0.00): https://e3nn.org/  (low trust (0.00): no positive signals matched)
- docs/curriculum/10-complexity-cognition/circuit-complexity.md → low-trust (0.00): https://rigetti.com/blog  (low trust (0.00): no positive signals matched)
- docs/curriculum/10-complexity-cognition/cognitive-architectures.md → low-trust (0.00): https://github.com/langchain-ai/langgraph  (low trust (0.00): no positive signals matched)
- docs/curriculum/10-complexity-cognition/cognitive-architectures.md → low-trust (0.00): https://github.com/langchain-ai/langgraph  (low trust (0.00): no positive signals matched)
- docs/curriculum/10-complexity-cognition/emergent-capabilities.md → low-trust (0.00): https://github.com/karpathy/nanoGPT  (low trust (0.00): no positive signals matched)
- docs/curriculum/10-complexity-cognition/emergent-capabilities.md → low-trust (0.00): https://github.com/EleutherAI/lm-evaluation-harness  (low trust (0.00): no positive signals matched)
- docs/curriculum/10-complexity-cognition/emergent-capabilities.md → low-trust (0.00): https://github.com/karpathy/nanoGPT  (low trust (0.00): no positive signals matched)
- docs/curriculum/10-complexity-cognition/complexity-classes.md → low-trust (0.00): https://github.com/prabakaranc98/FAIRE  (low trust (0.00): no positive signals matched)
- docs/curriculum/10-complexity-cognition/complexity-classes.md → low-trust (0.00): https://github.com/prabakaranc98/FAIRE  (low trust (0.00): no positive signals matched)
- docs/curriculum/08-causal-statistical-inference/disentanglement.md → low-trust (0.00): https://github.com/prabakaranc98/FAIRE  (low trust (0.00): no positive signals matched)
- docs/curriculum/08-causal-statistical-inference/disentanglement.md → low-trust (0.40): https://pyro.ai/examples/vae.html  (low trust (0.40): +0.40 approved-research-domain)
- docs/curriculum/08-causal-statistical-inference/do-calculus.md → low-trust (0.00): https://www.wiley.com/en-us/Causality%3A+Models%2C+Reasoning%2C+and+Inference-p-9780521895606  (low trust (0.00): no positive signals matched)
- docs/curriculum/08-causal-statistical-inference/counterfactuals.md → low-trust (0.00): https://www.basicbooks.com/titles/judea-pearl/the-book-of-why/9780465097609/  (low trust (0.00): no positive signals matched)
- docs/curriculum/08-causal-statistical-inference/counterfactuals.md → low-trust (0.00): https://github.com/uber/causalml  (low trust (0.00): no positive signals matched)
- docs/curriculum/08-causal-statistical-inference/counterfactuals.md → low-trust (0.00): https://github.com/uber/causalml  (low trust (0.00): no positive signals matched)
- docs/curriculum/08-causal-statistical-inference/counterfactuals.md → low-trust (0.00): https://github.com/py-why/dowhy  (low trust (0.00): no positive signals matched)
- docs/curriculum/15-ml-theory-foundations/entropy.md → low-trust (0.00): https://ieeexplore.ieee.org/document/6773024  (low trust (0.00): no positive signals matched)
- docs/curriculum/15-ml-theory-foundations/concentration.md → low-trust (0.40): https://arxiv.org/html/2605.13684  (low trust (0.40): +0.40 approved-research-domain)
- docs/curriculum/15-ml-theory-foundations/concentration.md → low-trust (0.40): https://arxiv.org/html/2510.02420v2  (low trust (0.40): +0.40 approved-research-domain)
- docs/curriculum/15-ml-theory-foundations/concentration.md → low-trust (0.40): https://arxiv.org/html/2510.02420v2  (low trust (0.40): +0.40 approved-research-domain)
- docs/curriculum/15-ml-theory-foundations/concentration.md → low-trust (0.40): https://pytorch.org/docs/stable/index.html  (low trust (0.40): +0.40 approved-research-domain)
- docs/curriculum/15-ml-theory-foundations/concentration.md → low-trust (0.40): https://jax.readthedocs.io/en/latest/  (low trust (0.40): +0.40 approved-research-domain)
- docs/curriculum/15-ml-theory-foundations/bias-variance.md → low-trust (0.00): https://netflixtechblog.com/  (low trust (0.00): no positive signals matched)
- docs/curriculum/15-ml-theory-foundations/bias-variance.md → low-trust (0.00): https://github.com/prabakaranc98/FAIRE  (low trust (0.00): no positive signals matched)
- docs/curriculum/15-ml-theory-foundations/bias-variance.md → low-trust (0.40): https://scikit-learn.org/stable/modules/model_selection.html  (low trust (0.40): +0.40 approved-research-domain)
- docs/curriculum/15-ml-theory-foundations/bias-variance.md → low-trust (0.00): https://lightning.ai/docs/pytorch/stable/  (low trust (0.00): no positive signals matched)
- docs/curriculum/03-representation-learning/bootstrapping-methods.md → low-trust (0.00): https://github.com/prabakaranc98/FAIRE  (low trust (0.00): no positive signals matched)
- docs/curriculum/03-representation-learning/bootstrapping-methods.md → low-trust (0.40): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html  (low trust (0.40): +0.40 approved-research-domain)
- docs/curriculum/03-representation-learning/bootstrapping-methods.md → low-trust (0.00): https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full  (low trust (0.00): no positive signals matched)
- docs/curriculum/03-representation-learning/contrastive-learning.md → low-trust (0.00): https://github.com/prabakaranc98/FAIRE  (low trust (0.00): no positive signals matched)
- docs/curriculum/09-algorithms-systems-ai/ai-hardware.md → low-trust (0.00): https://developer.nvidia.com/blog/nvidia-platform-delivers-lowest-token-cost-enabled-by-extreme-co-design/  (low trust (0.00): no positive signals matched)
- docs/curriculum/09-algorithms-systems-ai/ai-hardware.md → low-trust (0.00): https://dl.acm.org/doi/10.1145/3282506  (low trust (0.00): no positive signals matched)
- docs/curriculum/09-algorithms-systems-ai/ai-hardware.md → low-trust (0.00): https://github.com/vllm-project/vllm  (low trust (0.00): no positive signals matched)
- docs/curriculum/09-algorithms-systems-ai/ai-hardware.md → low-trust (0.00): https://github.com/NVIDIA/TensorRT-LLM  (low trust (0.00): no positive signals matched)
- docs/curriculum/09-algorithms-systems-ai/ai-hardware.md → low-trust (0.00): https://github.com/vllm-project/vllm  (low trust (0.00): no positive signals matched)

## Per-page summary

| Page | Kind | MVB variants | MVB pass | URLs | URL trusted |
|---|---|---|---|---|---|
| `docs/curriculum/07-attention-memory-reasoning/efficient-attention.md` | curriculum | 0 | 0 | 7 | 4 |
| `docs/curriculum/07-attention-memory-reasoning/single-head-attention.md` | curriculum | 0 | 0 | 5 | 4 |
| `docs/curriculum/07-attention-memory-reasoning/chain-of-thought.md` | curriculum | 0 | 0 | 10 | 10 |
| `docs/curriculum/05-statistical-probabilistic-ml/bayesian-inference.md` | curriculum | 0 | 0 | 5 | 5 |
| `docs/curriculum/05-statistical-probabilistic-ml/bayesian-nn.md` | curriculum | 0 | 0 | 3 | 2 |
| `docs/curriculum/05-statistical-probabilistic-ml/em.md` | curriculum | 0 | 0 | 6 | 1 |
| `docs/curriculum/04-neural-networks-dl/backpropagation.md` | curriculum | 0 | 0 | 12 | 9 |
| `docs/curriculum/04-neural-networks-dl/cnn.md` | curriculum | 0 | 0 | 13 | 8 |
| `docs/curriculum/14-biology-life-sciences/cell-simulation.md` | curriculum | 0 | 0 | 12 | 3 |
| `docs/curriculum/14-biology-life-sciences/epigenomics.md` | curriculum | 1 | 0 | 6 | 3 |
| `docs/curriculum/12-physics-scientific-ai/equivariant-networks.md` | curriculum | 0 | 0 | 11 | 9 |
| `docs/curriculum/10-complexity-cognition/circuit-complexity.md` | curriculum | 0 | 0 | 5 | 4 |
| `docs/curriculum/10-complexity-cognition/cognitive-architectures.md` | curriculum | 0 | 0 | 16 | 14 |
| `docs/curriculum/10-complexity-cognition/emergent-capabilities.md` | curriculum | 0 | 0 | 12 | 9 |
| `docs/curriculum/10-complexity-cognition/complexity-classes.md` | curriculum | 0 | 0 | 10 | 8 |
| `docs/curriculum/08-causal-statistical-inference/disentanglement.md` | curriculum | 0 | 0 | 5 | 3 |
| `docs/curriculum/08-causal-statistical-inference/do-calculus.md` | curriculum | 2 | 1 | 5 | 4 |
| `docs/curriculum/08-causal-statistical-inference/estimation.md` | curriculum | 3 | 2 | 5 | 5 |
| `docs/curriculum/08-causal-statistical-inference/counterfactuals.md` | curriculum | 0 | 0 | 13 | 9 |
| `docs/curriculum/15-ml-theory-foundations/entropy.md` | curriculum | 3 | 2 | 2 | 1 |
| `docs/curriculum/15-ml-theory-foundations/concentration.md` | curriculum | 0 | 0 | 15 | 10 |
| `docs/curriculum/15-ml-theory-foundations/bias-variance.md` | curriculum | 0 | 0 | 10 | 6 |
| `docs/curriculum/03-representation-learning/bootstrapping-methods.md` | curriculum | 0 | 0 | 11 | 8 |
| `docs/curriculum/03-representation-learning/contrastive-learning.md` | curriculum | 0 | 0 | 8 | 7 |
| `docs/curriculum/09-algorithms-systems-ai/ai-hardware.md` | curriculum | 0 | 0 | 14 | 9 |

