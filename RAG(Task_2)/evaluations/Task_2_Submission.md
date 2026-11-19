# Task 2: RAG from Scratch: Results

## Document and method

Corpus: NIST AI Risk Management Framework 1.0 (48 PDF pages). Retrieval uses `sentence-transformers/all-MiniLM-L6-v2` embeddings, normalized vectors, and a FAISS inner-product index. Evaluation uses 15 authored questions and counts a hit when the annotated physical PDF source page appears among the top five retrieved chunks.

## The 15 test questions and source passages

| # | Question | Printed page (PDF page) | Correct source passage |
|---:|---|---:|---|
| 1 | What is the primary purpose of the NIST AI Risk Management Framework? | 2 (7) | The goal of the AI RMF is to offer a resource to organizations designing, developing, deploying, or using AI systems to help manage AI risks and promote trustworthy and responsible development and use of AI systems. |
| 2 | What are the four core functions of the AI RMF? | 20 (25) | The AI RMF Core is composed of four functions: GOVERN, MAP, MEASURE, and MANAGE. |
| 3 | What is the purpose of the GOVERN function? | 21 (26) | The GOVERN function cultivates and implements a culture of risk management within organizations designing, developing, deploying, evaluating, or acquiring AI systems. |
| 4 | What is the purpose of the MAP function? | 24 (29) | The MAP function establishes the context to frame risks related to an AI system. |
| 5 | What is the purpose of the MEASURE function? | 28 (33) | The MEASURE function employs quantitative, qualitative, or mixed-method tools, techniques, and methodologies to analyze, assess, benchmark, and monitor AI risk and related impacts. |
| 6 | What is the purpose of the MANAGE function? | 31 (36) | The MANAGE function entails allocating risk resources to mapped and measured risks on a regular basis and as defined by the GOVERN function. |
| 7 | What are the characteristics of trustworthy AI systems? | 12 (17) | Trustworthy AI characteristics include valid and reliable, safe, secure and resilient, accountable and transparent, explainable and interpretable, privacy-enhanced, and fair with harmful bias managed. |
| 8 | Why is transparency important in AI systems? | 15 (20) | Transparency reflects the extent to which information about an AI system and its outputs is available to individuals interacting with such a system, whether or not they are aware they are doing so. |
| 9 | What does the AI RMF say accountability presupposes? | 15 (20) | Trustworthy AI depends upon accountability. Accountability presupposes transparency. |
| 10 | What does validation confirm for a specific AI system use? | 13 (18) | Validation is the confirmation, through objective evidence, that requirements for a specific intended use or application have been fulfilled. |
| 11 | How does the AI RMF define a safe AI system? | 14 (19) | AI systems should not, under defined conditions, lead to a state in which human life, health, property, or the environment is endangered. |
| 12 | Why is explainability important in trustworthy AI? | 16 (21) | Explainability and interpretability assist people operating or overseeing an AI system, as well as its users, to gain deeper insights into the system's functionality and trustworthiness, including its outputs. |
| 13 | Where does the AI RMF say human judgment should be used? | 12 (17) | Human judgment should be employed when deciding on the specific metrics related to AI trustworthiness characteristics and the precise threshold values for those metrics. |
| 14 | How does the AI RMF define risk tolerance? | 7 (12) | Risk tolerance refers to the organization's or AI actor's readiness to bear risk in order to achieve its objectives. |
| 15 | How should risk management be carried out over the AI system lifecycle? | 20 (25) | Risk management should be continuous, timely, and performed throughout the AI system lifecycle dimensions. |

## Retrieval accuracy and chunk-size choice

| Chunk size | Recall@5 | Hits |
|---:|---:|---:|
| 256 tokens | 66.67% | 10/15 |
| 512 tokens | 53.33% | 8/15 |

**The 256-token chunks worked better in this run**: they retrieved the annotated source page for 10/15 questions, compared with 8/15 for 512-token chunks. The shorter chunks likely helped by keeping each embedding focused on a narrower passage. This is a result from this 15-question set; the metric is page-level, so it does not prove that every returned chunk contained the exact supporting passage.

## Two retrieval failures and fixes

### Failure 1: What is the primary purpose of the NIST AI Risk Management Framework?

- Expected passage (printed page 2, PDF page 7): The goal of the AI RMF is to offer a resource to organizations designing, developing, deploying, or using AI systems to help manage AI risks and promote trustworthy and responsible development and use of AI systems.
- With 256-token chunks, top-five pages were: [1, 2, 37, 6, 8].
- Why it failed: the annotated source page did not appear in the top five.
- Fix: Add BM25 candidates for exact terms, retrieve a larger shortlist, then re-rank it.

### Failure 2: What does the AI RMF say accountability presupposes?

- Expected passage (printed page 15, PDF page 20): Trustworthy AI depends upon accountability. Accountability presupposes transparency.
- With 256-token chunks, top-five pages were: [47, 24, 8, 7, 7].
- Why it failed: the annotated source page did not appear in the top five.
- Fix: Add BM25 candidates for exact terms, retrieve a larger shortlist, then re-rank it.

## BM25 and re-ranking

BM25 adds keyword matching to embedding search. It helps when the question uses exact terms, acronyms, or rare names that semantic similarity may overlook. Combining BM25 candidates with vector results can improve coverage.

A re-ranker takes the query and candidate passages together, scores how relevant each passage is, and sorts the candidates again. It can improve the order of passages already retrieved, but cannot find a passage that did not enter the candidate set.
