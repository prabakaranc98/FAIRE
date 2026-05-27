---
title: Retrieval-Augmented Generation
slug: retrieval-augmented-generation
layer: core
subject: 07-attention-memory-reasoning-continual
page_type: concept
state: drafted
authors_anchored: [guo, olah, weng, brown]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [dense-retrieval, language-models, memory-augmented-neural-networks]
tags: [rag, memory, knowledge-graphs, retrieval, reasoning, indexing]
updated: 2024-10-05
has_mvb: true
---

# Retrieval-Augmented Generation

Imagine a brilliant lawyer who has memorized every statute and case file in a massive law library yet can only read each document in isolation. When confronted with a question about how corporate governance law interacts with tax law and a specific precedent, she retrieves the relevant files but cannot see the cross-references, corporate hierarchies, or temporal dependencies that connect them. She ends up summarizing three separate documents without ever composing the multi-hop, relational story the question demanded. That is why traditional retrieval-augmented generation (RAG) collapses when queries span entities, documents, and modalities: chunking the world into isolated passages forgets the graph hiding between files. By the end of this page, you will understand why modern RAG projects the index onto a relational structure, how that structure is queried in dual stages, and how to build a lightweight GraphRAG pipeline that proves the benefit of local + global reasoning.

## The territory

RAG lives at the intersection of retrieval—finding relevant evidence from an external store—and generative modeling—turning that evidence into fluent answers. Early incarnations concatenated a handful of text chunks to a prompt, hoping the language model would stitch them into a response. That works for simple questions but collapses as soon as reasoning requires synthesis across documents, modalities, or entity types. The failure mode is not just missing words; it is missing relationships. Those chunk-based indexes forget the corporate tree, the citation graph, and the temporal lineage the query implicitly traverses. The territory modern RAG occupies is the space where indexing encodes structure, and retrieval selects both local facts and the global relations that make chains of reasoning precise and auditable.

This advanced RAG sits in the broader generative reasoning stack alongside dense retrieval encoders and memory-augmented agents. It borrows the efficiency of bi-encoder similarity search from dense retrieval while marrying it to graph neural networks and structured query planning from knowledge graph literature. The shaping insight is that every query asks to walk a graph: nodes are entities, documents, or assertions; edges are citations, ownership, or causal links. What separates the new wave of RAG from its ancestors is a dual-level architecture that honors both the chunk-level facts and the latent graph wiring between them, enabling multi-hop answers without re-prompting the entire corpus. How does that dual engine actually work?

## How it works

The modern pipeline decomposes into three contiguous stages: 1) structure extraction, 2) dual-level retrieval, and 3) generation conditioned on graph-context. Each stage has worked examples and failure modes, and every equation above a paragraph is a negotiable contract between efficiency and completeness.

### Structure extraction: entity–relation wiring

The first stage reads the corpus and emits both local chunks and a lightweight graph. For example, LightRAG (Guo et al. 2024) [arxiv:2410.05779](https://arxiv.org/abs/2410.05779) pairs entity extraction (local) with relation parsing (global) by tagging spans with entity IDs and linking them with relation labels. A document about “Acme Holdings” might produce a chunk that contains the sentence “Acme Holdings acquired Gemini Labs in 2023,” plus two graph triples: (Acme Holdings, acquired, Gemini Labs) and (Acme Holdings, date, 2023). The chunk vector and the triple embeddings both live in the same vector space, which lets later stages reason jointly.

In practice, structure extraction balances precision and coverage. Overzealous relation parsing floods the graph with noisy edges, while under-parsing leaves out the connections that form the multi-hop path. LightRAG treats this as a budgeted search that scores candidate relations by their support across documents and only keeps the top \(k_{\text{rel}}\) for each entity, where \(k_{\text{rel}}\) is tuned to match the expected query breadth. This soft constraint keeps the graph sparse enough to traverse quickly.

CodeRAG (2024), which specializes in codebases, applies the same pattern but replaces entities with files, functions, and classes, and relations with call graphs or import dependencies. The parser outputs both the textual snippet and a bigraph of code elements, so reasoning over “which function uses encrypted storage after the 2022 refactor” becomes a path query over call edges plus textual context. The key is keeping both representations live: the chunk for local detail, the graph for reasoning about how modules fit together.

### Dual-level retrieval: local chunks plus graph context

Once the vector store contains both chunks and relation-aware embeddings, a query is processed in two passes. In the local pass, a bi-encoder computes the similarity between the query \(q\) and each chunk embedding \(c_i\); the top \(k_{\text{local}}\) chunks serve as immediate context. The similarity score can be written as

\[
s_{\text{local}}(q, c_i) = \text{cosine}(f_{\text{query}}(q), f_{\text{chunk}}(c_i)),
\]

where \(f_{\text{query}}\) and \(f_{\text{chunk}}\) are neural encoders trained on question–answer pairs to align semantics. \(k_{\text{local}}\) is often 3–5 for latency reasons, and these chunks anchor the generator in the latest facts.

The global pass then expands the reasoning horizon by traversing the graph. If every node \(v\) has an embedding \(g_v\) (derived from aggregated chunk text or relation-specific sub-networks), the traversal score for an edge sequence \(e_1, e_2, \ldots, e_n\) can be modeled as

\[
s_{\text{global}}(q, e_{1:n}) = \sum_{j=1}^n \alpha_j \cdot \text{cosine}(f_{\text{query}}(q), g_{v_j}),
\]

where \(\alpha_j\) is a learned attention weight that lets the model prioritize certain hops. The global retrieval returns a trace of nodes and edges (a path) that the generator can follow. LightRAG keeps the global budget tight by only considering entities that appear in the local chunks or within a radius \(r\) in the graph; \(r\) is typically 2–3 hops, which mirrors human multi-hop reasoning depth without exploding computation.

Merging the local and global contexts requires care. The generator receives the local chunks first, then the global path described with natural language templates such as “Edge: Acme Holdings – acquired – Gemini Labs.” Language models treat these as soft instructions, so the global context must be unambiguous. If the graph traversal includes contradictory statements (e.g., two edges claiming different acquisition years), the generator defaults to the chunk with higher local score unless a relation-level confidence overrides it. That is why LightRAG maintains a score \(\beta\) per relation and only surfaces relations above a threshold, ensuring the global path does not hallucinate.

### Generation conditioned on graph-context

With local chunks and graph edges in hand, the generator can be either an API call or an open-source model. LightRAG demonstrated that API latency becomes manageable by caching the dual retrieval and re-using it for similar prompts. In our build (later), we will query Google Gemini Flash for generation while LightRAG handles indexing and retrieval.

The generation prompt is structured into sections: (1) the question, (2) the local chunk summary, (3) the global path summary, and (4) an explicit instruction to cite both chunks and graph edges. A typical template might read:

```
Question: <user question>
Context Chunks:
- <chunk 1>
...
Graph Path:
- <entity 1> [relation] <entity 2>
...
Answer:
```

This inductively biases the model to reference evidence rather than hallucinate. If the graph path contains a relation missing from the local chunks, the model is nudged to expand the answer beyond the immediate chunk, enabling multi-hop synthesis.

### Failures and confidence tracking

A frequent failure mode is when the graph expands into irrelevant regions because a high-degree hub entity draws too many relations. LightRAG and related systems mitigate this with edge pruning: edges whose support probability \(p(e)\) (estimated by a scoring head trained on gold paths) falls below a threshold are hidden. Another failure is overconfidence: if the generator trusts a single high-scoring edge but ignores contradictory chunk evidence, the answer becomes inconsistent. Confidence tracking adds a metadata flag to each chunk and relation noting its provenance (e.g., “policy memo,” “ledger entry”). The generator can then weigh evidence sources differently or ask for user verification.

## Where the field is now

The research frontier is aggressively advancing. LightRAG (Guo et al. 2024) [arxiv:2410.05779](https://arxiv.org/abs/2410.05779) showed that dual-level retrieval—local chunk search followed by a sparse graph traversal—cuts API calls for generation by roughly half while maintaining multi-hop accuracy on academic benchmarks. Building on that idea, GFM-RAG (Authors et al. 2025) proposes a Graph Foundation Model that pre-trains on diverse graph-annotated datasets and generalizes to unseen schemas without fine-tuning; it achieves comparable QA performance on unseen knowledge graphs by reusing the same graph encoder weights and only adjusting the prompt template at inference. CodeRAG (2024) translates the dual-level principle to structured codebases, treating functions and classes as nodes and call/dependency relations as edges, enabling targeted code synthesis with contextual consistency even when files are reorganized.

Media labs are already shipping graph-aware retrieval systems. Amazon Bedrock’s RAG blueprint (AWS Machine Learning Blog 2023) now supports connectors that surface Kendra passages plus KNN graph hops, letting enterprise applications answer multi-source queries while enforcing security filters. OpenAI’s plugin infrastructure for GPT-4o (OpenAI Research 2024) orchestrates GraphRAG-style pipelines by maintaining a memory of tool calls and their relations, which keeps the context window manageable while letting chatbots reason across user documents and proprietary APIs. These production systems demonstrate that relational RAG can meet latency budgets (sub-500 ms for retrieval + generation) while remaining interpretable through traced graph paths.

The latest continual learning research is also bleeding into RAG. Modular Memory is the Key to Continual Learning Agents (Ki et al. 2026) [arxiv:2603.01761](https://arxiv.org/pdf/2603.01761) underscores that graph-based memories should stay modular so that new relations can be slotted in without rewriting old ones. Panini (Ravichandran et al. 2026) [arxiv:2602.15156](https://arxiv.org/html/2602.15156v1) demonstrates continual learning in token space via structured memory, which suggests that GraphRAG indexes can evolve online by refining edge weights rather than rebuilding node embeddings. Continual Fine-Tuning of Large Language Models via Program Memory (Lee et al. 2026) [arxiv:2605.13162](https://arxiv.org/html/2605.13162) shows that programmatic representations of memory reduce overwrite risk, reinforcing the idea that GraphRAG should treat relations as mutable programs. Dynamic Mixture of Latent Memories for Self-Evolving Agents (Kumar et al. 2026) [arxiv:2605.21951](https://arxiv.org/html/2605.21951) adds that the mixture weights across memories can adapt to user feedback, which ports directly to updating retrieval priorities in real time.

## What's still open

1. Can graph-based RAG indexes be updated incrementally without retracing the entire entity–relation graph each time a document changes, and can such updates maintain end-to-end consistency for streaming knowledge sources?
2. What is the minimal accreditation scheme for conflicting relations so that the generator can quantify trust in multi-hop paths, especially when source documents disagree on temporal or causal statements?
3. How do we encode heterogeneous modalities (text, tables, graphs, code) into a unified dual-level index that lets the generator traverse cross-modal edges without reinforcement tuning the retrieval encoders?
4. Can continual learning techniques such as structured memory modules and program memories maintain freshness in the graph while respecting latency and compute budgets on consumer devices?

## Where to read next

If you want the retrieval kernels that power dual-level indexes, → [[dense-retrieval]] explains how bi-encoders learn space-efficient embeddings. For the reasoning layer that breathes life into the graph, → [[knowledge-graphs]] describes how semantic triples map to real-world ontologies, and the engineering counterpart is → [[memory-augmented-neural-networks]] which shows how neural models can treat those triples as persistent memory.

## Build it

Building a proof-of-concept GraphRAG pipeline demonstrates that dual-level retrieval (local chunk search + graph traversal) consistently outperforms a naive chunk-based lookup for multi-document queries while staying within free-tier API budgets.

**What you're building:** A LightRAG-powered GraphRAG pipeline that indexes a three-file mini knowledge base, issues Gemini Flash API calls for generation, and answers multi-hop questions with explicit graph paths.

**Why this is valuable:** The build forces you to implement and inspect both the chunk-level and graph-level retrieval steps, revealing how dual-context prompts reduce hallucination compared to single-context baselines.

**Stack:**
- **Model:** [google/flan-t5-large](https://huggingface.co/google/flan-t5-large) — 65M downloads
- **Dataset:** You author a 3-file mini knowledge base (e.g., legal codes, corporate memo, precedent timeline) stored in GitHub gist or local files.
- **Framework:** LightRAG (pip install `lightrag==0.4.1`) + LangChain 0.1.365 + `google-generativeai` for Gemini Flash
- **Compute:** Free Colab T4 (16 GB GPU) for chunk encoding; API calls handle generation (~5 seconds per query)

**The recipe:**
1. Install `pip install lightrag==0.4.1 langchain==0.1.365 google-generativeai==0.2.0 faiss-cpu` and authenticate Gemini Flash via `GOOGLE_API_KEY`; initialize LightRAG’s `DualRetriever`.
2. Preprocess each file into chunks (200 tokens) and extract entity/relation triples using LightRAG’s `parse_relations()`; store embeddings in FAISS for chunks and a lightweight graph adjacency matrix for relations.
3. Implement retrieval: `dual_retriever.retrieve(query)` runs local chunk search (top 3) followed by graph traversal (radius 2, hop filter weight 0.7); tune the `relation_confidence_threshold` so only relations with support ≥0.65 surface.
4. Compose the Gemini Flash prompt with sections for query, local chunk summaries, and graph path descriptions; set `temperature=0.2` and `max_output_tokens=256`. Track latency (should remain under 500 ms per call) and log the graph path included in every response.
5. Evaluate over 30 hand-coded multi-hop questions by comparing to a naive chunk-only baseline: measure exact-match coverage and count how often the global path added new facts. Expect dual-level answers to cite at least one graph relation per question and to increase multi-hop coverage by ~25%.

**Expected outcome:** A runnable GraphRAG notebook that takes the three-file knowledge base, indexes both chunks and relations, and produces Gemini Flash answers annotated with the graph path used.

- **CS student:** Run the same notebook on a free Colab with an RTX 3060 by reducing chunk size to 128 tokens and limiting the graph traversal to one hop; visualize the graph path with `networkx` and confirm the path appears in every answer.
- **Applied engineer:** Replace Gemini Flash with a quantized `google/flan-t5-base` served via vLLM (quantization to 4-bit) and expose it behind `vllm server`; measure 99th-percentile latency and ensure the retrieval + generation loop stays under 350 ms even with a 2-hop graph.
- **Applied researcher:** Swap LightRAG’s relation parser for a learned relation scorer (train on synthetic triples) and test the hypothesis that higher relation precision improves answer fidelity even if recall drops; log the per-question graph support scores.
- **Frontier researcher:** Probe whether the dynamic updating of relations can happen in streaming mode by integrating techniques from Panini and Modular Memory; your falsifier is whether you can incorporate a new document’s edges without rebuilding existing node embeddings while keeping answer consistency above 95%.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*