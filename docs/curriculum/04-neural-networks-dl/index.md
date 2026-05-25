---
title: Neural Networks & Deep Learning
tags: [neural-networks, deep-learning, optimization, backpropagation, architectures]
---

# Track 04 · Neural Networks & Deep Learning

> The mechanics of learning: architectures, optimization, training dynamics, and the scaling laws that govern modern deep learning.

This track covers how neural networks actually work — from the mathematics of backpropagation to the empirical science of training large models. It is the engineering substrate everything else is built on.

---

## Topics

### Foundations
- [Perceptrons & MLPs](mlp.md) — universal approximation, activation functions, feedforward networks
- [Backpropagation](backpropagation.md) — computational graphs, chain rule, gradient flow
- [Optimization](optimization.md) — SGD, Adam, learning rate schedules, loss landscapes

### Architectures
- [Convolutional Neural Networks](cnn.md) — convolution, pooling, receptive fields, modern CNN designs
- [Recurrent Neural Networks](rnn.md) — LSTM, GRU, sequence modeling, vanishing gradients
- [Residual Networks](residual-networks.md) — skip connections, Highway Networks, depth and optimization

### Training Dynamics
- [Normalization](normalization.md) — BatchNorm, LayerNorm, RMSNorm, their roles in training stability
- [Regularization](regularization.md) — dropout, weight decay, data augmentation, early stopping
- [Scaling Laws](scaling-laws.md) — Chinchilla, compute-optimal training, emergent capabilities

### Modern Deep Learning
- [Hyperparameter Tuning](hyperparameter-tuning.md) — grid search, Bayesian optimization, random search
- [Transfer Learning](transfer-learning.md) — fine-tuning, feature extraction, domain adaptation

---

## Connections to frontier research

- **Training at scale** — the engineering of large model training: mixed precision, gradient checkpointing, ZeRO
- **Mechanistic interpretability** — understanding what neural networks actually compute
- **Neural collapse** — terminal phase of training dynamics and its implications
- **Grokking** — delayed generalization, phase transitions, and double descent

---

## Recommended entry points

Start with [Perceptrons & MLPs](mlp.md) and [Backpropagation](backpropagation.md), then [Optimization](optimization.md). For frontier relevance, go straight to [Scaling Laws](scaling-laws.md).
