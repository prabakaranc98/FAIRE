```yaml
---
title: Search Algorithms
track: 01-ai
tags: [information retrieval, semantic search, vector search, embeddings, reasoning]
depth: foundations
prereqs: [natural-language-processing, vector-embeddings]
updated: 2024-10-26
has_mvb: false
---

# Search Algorithms
> **TL;DR:** Search algorithms are evolving from simple keyword matching to incorporate semantic understanding and reasoning, enabling more accurate and contextually relevant information retrieval.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms--techniques) → [In production](#in-production) | Understand real-world applications |
| Curious generalist | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) | Grasp the big picture |
| Math/theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Understand the underlying principles |
| Researcher / frontier | [Current SotA](#current-sota) → [Seminal papers](#seminal-papers--test-of-time) | Identify research opportunities |

---

## What it is
Imagine you're trying to find the perfect hiking boots online. You type "waterproof boots, size 10, good for rocky trails" into the search bar. Behind the scenes, a search algorithm must sift through millions of products, understand your nuanced request, and deliver the most relevant results. This process, seemingly simple, is a complex dance of data retrieval and interpretation.

Traditional search algorithms rely heavily on keyword matching, which can often lead to irrelevant results. Modern search algorithms, however, are evolving to incorporate semantic understanding, reasoning capabilities, and contextual awareness. These advancements allow search engines to better interpret user intent and provide more accurate and personalized search results.

The shift towards semantic search involves techniques like vector embeddings, which represent words and phrases as numerical vectors capturing their meaning. By comparing the vector representations of queries and documents, search algorithms can identify semantically similar content even if the exact keywords don't match. This evolution is crucial for handling the increasing complexity and volume of information available online.

## Why it matters at the frontier
Search algorithms are a critical component of many frontier AI applications, including question answering systems, research assistants, and autonomous agents. The ability to efficiently and accurately retrieve relevant information is essential for these systems to perform complex tasks and make informed decisions. Improving search algorithms directly impacts the performance and reliability of these advanced AI systems.

The development of search algorithms that can effectively handle conflicting or noisy search results, particularly in reasoning-intensive queries, is an open problem. Addressing this challenge is crucial for improving the accuracy of search-augmented language models and enabling them to tackle more complex reasoning tasks. This is a key area of focus for researchers aiming to build more robust and reliable AI systems.

## Core concepts
- **Keyword matching** — A traditional search technique that identifies documents containing the exact keywords specified in a query.
- **Semantic search** — A search approach that aims to understand the meaning and context of a query to retrieve more relevant results.
- **Vector embeddings** — Numerical representations of words, phrases, or documents that capture their semantic meaning.
- **Vector database** — A database optimized for storing and querying vector embeddings, enabling efficient similarity searches.
- **Information retrieval** — The process of obtaining relevant information from a collection of resources.
- **Relevance ranking** — The process of ordering search results based on their relevance to the user's query.
- **Search-augmented language model** — A language model that uses search algorithms to retrieve external information to improve its performance on tasks such as question answering and reasoning.

## Mathematical foundations
While the specific equations vary depending on the algorithm, a core concept is similarity scoring using vector embeddings. A common approach is cosine similarity:

\[
\text{similarity}(q, d) = \frac{\mathbf{v}_q \cdot \mathbf{v}_d}{\|\mathbf{v}_q\| \|\mathbf{v}_d\|}
\]

where \(\mathbf{v}_q\) is the vector embedding of the query, \(\mathbf{v}_d\) is the vector embedding of the document, \(\mathbf{v}_q \cdot \mathbf{v}_d\) is the dot product of the two vectors, and \(\|\mathbf{v}_q\|\) and \(\|\mathbf{v}_d\|\) are the magnitudes of the vectors. This equation calculates the cosine of the angle between the query and document vectors, providing a measure of their similarity. The closer the cosine similarity is to 1, the more similar the query and document are.

Another important concept is the use of loss functions to train embedding models. For example, a contrastive loss might be used:

\[
L = \sum_{i=1}^{N} y_i \cdot d(\mathbf{a}_i, \mathbf{b}_i) + (1 - y_i) \cdot \max(0, m - d(\mathbf{a}_i, \mathbf{b}_i))
\]

where \(L\) is the loss, \(y_i\) is a binary label indicating whether the pair is similar (1) or dissimilar (0), \(\mathbf{a}_i\) and \(\mathbf{b}_i\) are the embeddings of the pair, \(d(\mathbf{a}_i, \mathbf{b}_i)\) is the distance between the embeddings, and \(m\) is a margin. This loss function encourages similar pairs to have small distances and dissimilar pairs to have distances greater than the margin.

## Key algorithms / techniques
- **TF-IDF (Term Frequency-Inverse Document Frequency)** — A traditional information retrieval technique that weighs terms based on their frequency in a document and their rarity across the corpus.
- **BM25 (Best Matching 25)** — An extension of TF-IDF that incorporates document length normalization and term saturation to improve relevance ranking.
- **Word2Vec** — A neural network-based technique for learning word embeddings by predicting the context words surrounding a given word.
- **GloVe (Global Vectors for Word Representation)** — A word embedding technique that leverages global word co-occurrence statistics to learn vector representations.
- **Sentence Transformers** — A family of transformer-based models specifically designed for generating high-quality sentence embeddings.

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent | 2025 | Zhang et al. | Introduces a new benchmark for evaluating deep-research agents, disentangling retrieval from reasoning. |
| ReasonIR: Learning to Reason with Search for LLMs via Reinforcement Learning | 2025 | Shao et al. | Explores reinforcement learning for reasoning-search interleaved LLM agents. |
| Punctuated Equilibria in Artificial Intelligence: The Institutional Scaling Law and the Speciation of Sovereign AI | 2026 | Baciak et al. | Explores the historical development of AI and representative philosophical thinking from the perspective of punctuated equilibria. |

## Seminal papers & test-of-time
| Paper | Year | Key contribution |
|---|---|---|
| On the Origin of Deep Learning | 2017 | Wang et al. | Reviews the evolutionary history of deep learning models. |
| Artificial Intelligence: 70 Years Down the Road | 2023 | Lin Zhang | Summarizes the development of AI over the past 70 years. |
| Turing’s Test, a Beautiful Thought Experiment | 2024 | Gonçalves | Discusses the Turing test in the context of recent AI developments. |

## Current SotA
Zhang et al. (2025) introduced BrowseComp-Plus, a new benchmark for evaluating deep-research agents, disentangling retrieval from reasoning. Shao et al. (2025) explored reinforcement learning for reasoning-search interleaved LLM agents, analyzing the impact of reward design, LLM backbone, and search engine choice.

## What's happening now
Research is actively exploring how to integrate search with large language models (LLMs) to enhance their reasoning and knowledge capabilities. This includes developing new architectures and training techniques that allow LLMs to effectively leverage external information retrieved from search engines. The goal is to create more powerful and reliable AI systems that can tackle complex tasks requiring both reasoning and access to up-to-date information.

Engineering efforts are focused on building scalable and efficient search systems that can handle the increasing volume and complexity of data. This includes developing new vector database technologies and optimizing search algorithms for performance and accuracy. Companies are also exploring how to integrate search into their products and services to improve user experience and provide more personalized results.

A key open problem is: How can we develop search algorithms that effectively handle conflicting or noisy search results, particularly in reasoning-intensive queries, to improve the accuracy of search-augmented language models? Addressing this challenge is crucial for building more robust and reliable AI systems that can leverage search to solve complex problems.

## In production
- Databricks — Building real-time product search at scale using Databricks Vector Search — Handles ingestion, retrieval, and refinement in one platform — [https://www.databricks.com/blog/building-real-time-product-search-databricks]
- Ibotta — Powers real-time and batch machine learning within its mobile search engine — At scale — [https://aws.amazon.com/blogs/machine-learning/powering-a-search-engine-with-amazon-sagemaker/]
- Tyson Foods — AI-powered conversational assistant for customer search — Scalable, production-grade architecture on AWS — [https://aws.amazon.com/blogs/machine-learning/tyson-foods-elevates-customer-search-experience-with-an-ai-powered-conversational-assistant/]
- Google Research — SOAR (Spilling with Orthogonality-Amplified Residuals) for ScaNN — Improves vector index with controlled redundancy — [https://research.google/blog/soar-new-algorithms-for-even-faster-vector-search-with-scann/]

## Code & implementations
*For a hands-on build with this concept, see the MVB on [[vector-embeddings]].*

## What comes next

- [[vector-embeddings]] — provides the foundation for representing text as numerical vectors, enabling semantic similarity calculations.
- [[question-answering]] — leverages search algorithms to retrieve relevant information for answering user questions.
- [[retrieval-augmented-generation]] — combines search with language models to generate more informative and contextually relevant text.

## Connected topics

- [Classical Planning](./classical-planning.md) — Classical planning algorithms are a type of search algorithm.
- [Agent Architectures](./agent-architectures.md) — Agent architectures often incorporate search algorithms for decision-making.
- [Markov Decision Process](../06-reinforcement-learning/mdp.md) — MDPs often use search algorithms to find optimal policies.
- [Proximal Policy Optimization (PPO)](../06-reinforcement-learning/ppo.md) — PPO is a reinforcement learning algorithm that uses search techniques.
- [Reinforcement Learning from Human Feedback (RLHF)](../06-reinforcement-learning/rlhf.md) — RLHF uses reinforcement learning, which often relies on search algorithms.
- [Cognitive Architectures](../10-complexity-cognition/cognitive-architectures.md) — Cognitive architectures may employ search algorithms for problem solving.


## Further reading
- Wang et al. (2017) — "On the Origin of Deep Learning" — [https://arxiv.org/pdf/1702.07800] — Reviews the evolutionary history of deep learning models, providing context for the development of modern search algorithms.
- Lin Zhang (2023) — "Artificial Intelligence: 70 Years Down the Road" — [https://arxiv.org/pdf/2303.02819] — Summarizes the development of AI over the past 70 years, highlighting the role of search in various AI applications.
- Gonçalves (2024) — "Turing’s Test, a Beautiful Thought Experiment" — [https://arxiv.org/abs/2401.00009] — Discusses the Turing test in the context of recent AI developments, including the use of search in AI systems.
- Baciak et al. (2026) — "Punctuated Equilibria in Artificial Intelligence: The Institutional Scaling Law and the Speciation of Sovereign AI" — [https://arxiv.org/pdf/2603.14664] — Explores the historical development of AI and representative philosophical thinking from the perspective of punctuated equilibria.
```