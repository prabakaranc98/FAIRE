---
skill: math-latex
description: Write mathematical content correctly for MkDocs + MathJax rendering
applies_to: [write_draft, revise_draft, scratch]
triggers: [equation, LaTeX, formula, derivation, proof, loss function, objective, distribution]
---

# Skill: Mathematical Notation

## Rendering environment

This wiki uses MkDocs Material + pymdownx.arithmatex + MathJax 3.
The arithmatex extension only wraps `\[...\]` and `\(...\)` delimiters.
**Dollar signs (`$...$`, `$$...$$`) do NOT render.** Never use them.

## Delimiters

| Context | Correct | Wrong (won't render) |
|---|---|---|
| Inline math | `\(x_t\)` | `$x_t$` |
| Display (block) | `\[\mathcal{L} = ...\]` | `$$\mathcal{L} = ...$$` |

## Equation structure

Every displayed equation must be followed by an annotation line:

```
\[
q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t)I\right)
\]

where \(\bar{\alpha}_t = \prod_{s=1}^{t}(1 - \beta_s)\) is the cumulative noise schedule,
\(x_0\) is the clean data sample, and \(I\) is the identity matrix.
```

**Rules:**
1. Annotate EVERY symbol on the line immediately after the equation
2. Use the `where ... is ..., ... is ...` pattern
3. Give intuition after the annotation: "This equation says that..."
4. Never leave a bare equation without annotation
5. Max 3–5 equations per page; choose the ones that genuinely clarify

## Variable naming conventions

- Scalars: `\(x\)`, `\(t\)`, `\(\alpha\)`
- Vectors: `\(\mathbf{x}\)`, `\(\boldsymbol{\mu}\)`
- Matrices: `\(\mathbf{W}\)`, `\(\mathbf{H}\)`
- Sets: `\(\mathcal{X}\)`, `\(\mathcal{N}\)`, `\(\mathcal{L}\)`
- Expectations: `\(\mathbb{E}_{x \sim p}[f(x)]\)`
- Distributions: `\(p(x \mid \theta)\)`, `\(\mathcal{N}(\mu, \sigma^2)\)`

## Common pitfalls

- `\left(` and `\right)` for auto-sizing brackets
- `\mid` for conditional probability (not `|`)
- `\text{KL}` for named operators, not `KL`
- `\,` for thin spaces in integrals and products
- Align multi-line derivations with `align` environment inside `\[...\]`
