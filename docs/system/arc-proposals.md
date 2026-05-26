```markdown
---
title: Arc proposals
generated_at: 2024-05-26 14:00 UTC
remaining_budget: $3.44
slots_open: 2
---

# Arc proposals

This cycle proposes one arc focused on JEPA world models, aiming to build a predictive model for action. The survey revealed promising directions in Gaussian Splatting, JEPA, and Diffusion-based world models. After careful evaluation, the JEPA arc was selected due to its strong compounding trajectory and alignment with existing curriculum. The other branches were either not diagonal or had insufficient curriculum support.

## 1. JEPA World Model for Action — EV/$ = 4.5 — verdict: approve
**Destination:** Build a JEPA-style world model and use it to predict future states conditioned on actions.
**Steps:** 6 · **Cost:** $1.60 · **Impact:** 8/10
**Prereqs in curriculum:** transformer architecture ✓(07-attention-memory-reasoning), representation learning ✓(03-representation-learning), optimization ✓(04-neural-networks-dl), causal inference ✗ (queue first)
**Persona span:** 3 (curious-learner, applied-researcher, frontier-researcher)
**Seminal anchors:** LeCun 2022 · V-JEPA 2024 ·  Schrittwieser et al. 2020
**Outline:**
1.  **Build:** Implement a simple autoencoder (representation learning)
    *   `mvb_persona`: curious-learner
2.  **Build:** Implement a Transformer-based predictor (transformer architecture)
    *   `mvb_persona`: curious-learner
3.  **Build:** Train the predictor to predict future frames (representation learning, optimization)
    *   `mvb_persona`: applied-researcher
4.  **Build:** Add action conditioning to the predictor (transformer architecture)
    *   `mvb_persona`: applied-researcher
5.  **Build:** Evaluate the model's ability to predict future states given actions.
    *   `mvb_persona`: applied-researcher
6.  **Build:** Explore the model's latent space for planning and control.
    *   `mvb_persona`: frontier-researcher
**Editor verdict:** approve
**Approval note:** This arc provides a solid foundation for understanding and building JEPA-style world models, a promising direction for future AI systems. The compounding trajectory is clear, and the persona walk is well-structured.

## 2. Exploration deferred — seed "world models"

The survey returned 3 distinct branching directions. The Gaussian Splatting and Diffusion-based world model branches were considered, but were not selected due to the lack of a clear diagonal shape, or insufficient curriculum support.
```