# Evaluation

## Methodology

Development proceeded through three progressive configurations, each
building on the previous one's limitations, tested against the same 20-query
set spanning five categories: registration/study, exams & grading,
academic warnings/dismissal, graduation requirements, and out-of-scope
queries (used to test refusal behavior).

| Configuration | Description | Factual Correctness | Mean Latency |
|---|---|---|---|
| A — Retrieval only | Baseline, no generative model | 55% (11/20) | 12.5s |
| B — Enhanced retrieval | Multiple phrasing variants per question in the KB | 90% (18/20) | 12.95s |
| C — Full RAG + fine-tuned LLM | Qwen3-8B + LoRA on top of retrieval | 65%* (13/20) | 21s |

*Configuration C trades some raw factual-match score for more natural,
well-formed Arabic answers versus verbatim-retrieved text; this is a known
trade-off discussed further below.

**Generation/retrieval parameters (fixed across all configs):**

| Parameter | Value |
|---|---|
| Temperature | 0.2 |
| Max new tokens | 512 |
| Top-p | 0.9 |
| Repetition penalty | 1.15 |
| Retrieval k | 3 |
| Similarity threshold (cosine distance) | 1.5 |

## Quantitative results (final system)

BERTScore against ground-truth answers:

- Precision: 0.701
- Recall: 0.732
- **F1: 0.715**

## User satisfaction survey (n=16 students, 5-point Likert scale)

| Dimension | Mean score |
|---|---|
| Response Clarity | 4.25 |
| Response Relevance | 4.5 |
| Response Completeness | 3.75 |
| Response Speed Acceptability | 3.4 |
| Perceived Ease of Use | 4.75 |
| **Overall Satisfaction** | **3.9** |

100% of participants said they'd use the chatbot again.

## Discussion — what the numbers actually mean

- **Ease of use and relevance scored highest** — the RAG grounding
  approach works: students get answers tied to real regulation text.
- **Completeness scored lower (3.75)** — the strict "don't add information
  not in the retrieved context" prompting reduces hallucination risk but
  can make answers feel less elaborated. This is a deliberate trade-off,
  not a bug.
- **Response speed (3.4)** is the weakest dimension, directly caused by
  running inference inside a shared Google Colab GPU runtime rather than a
  dedicated inference server.

## Model selection rationale

Nine LLMs were qualitatively screened (Llama 3.1 8B/16-bit, Jais-2 70B,
Allam-2 7B, EuroLLM 22B, SILM V3, Qwen3 8B/30B, AceGPT 13B) on Arabic
quality, hallucination rate, instruction following, Sudanese dialect
support, and latency. **Qwen3-8B** was selected as the best balance of
response quality vs. computational cost — it was the only model in this
range with strong Sudanese dialect handling and low hallucination at a
deployable size.

## Limitations (acknowledged, not hidden)

1. Knowledge base scope is limited to two official documents; some genuine
   student questions fall outside coverage and receive a "no reliable
   information" refusal rather than a guess — this is intentional but
   means real information gaps remain unresolved for those students.
2. Fine-tuning dataset (~200 pairs) is small, carrying overfitting risk;
   mitigated via LoRA dropout, gradient accumulation, and the fact that
   factual grounding relies on retrieval, not the fine-tuned weights alone.
3. Arabic response quality assessment involved a 2-evaluator panel; some
   subjectivity in judging register/naturalness is unavoidable.
4. Inference latency is tied to shared Colab GPU availability and is not
   representative of a production-grade dedicated server.
