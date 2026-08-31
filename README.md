# Language Predictability & Model Collapse Visualizer

This repository provides a PyTorch and Matplotlib framework designed to analyze, measure, and visualize text predictability ("surplexity") and long-range information entropy across diverse text domains using causal language models such as `Qwen/Qwen2.5-1.5B`, `OLMo 2`, and `Llama 3.2`.

---

## Theoretical Background

The visualizer operationalizes principles from two theoretical frameworks in language modeling and statistical physics:

### 1. Learning by Surprise (Gambetta et al., 2026)

* **Model Collapse Mechanism:** When generative language models are recursively fine-tuned on synthetic, AI-generated content (AI Autophagy), output distributions degenerate. Training on low-perplexity, highly predictable tokens causes next-token probability distributions to over-concentrate (high Gini coefficient), accelerating model collapse and eroding lexical diversity.
* **Surprise as an Information Filter:** Measuring token predictability during data selection and prioritizing high-perplexity ("surprising") passages reintroduces distribution variance. Selecting samples with high surprise acts as an un-annotated filter, preserving reasoning capabilities and long-tail domain knowledge without needing human vs. synthetic data labels.

### 2. Information Entropy & Context Scaling (Scheibner et al., 2026)

* **Continuous Long-Range Entropy Decay:** Early estimations of written English entropy suggested that conditional entropy reached a plateau after $N \sim 100$ characters. Modern LLMs demonstrate that mean code length $L(N)$ continues to decrease logarithmically out to at least $N \sim 10^4$ characters without reaching a plateau in standard prose. This continuous decay indicates effective direct interactions and dependencies across thousands of characters.


* **Emergent Certainty at Large Scale:** Evaluating the distribution of conditional entropy $P(\mathcal{S}_{\text{cond}})$ shows that as context length $K$ grows, a sharp peak accumulates near zero entropy ($\mathcal{S}_{\text{cond}} \to 0$) alongside a power-law tail at small code lengths. Long contexts create pockets of near-absolute predictability for specific tokens.


* **Learning Dynamics Across Context Scales:** Models master local grammatical and spelling rules ($N < 100$) rapidly during early pre-training, but acquire long-ranged contextual dependencies ($N > 1000$) gradually over extended training steps.


* **Genre-Dependent Limits:** Scaling behavior varies by genre. Structured prose (C4, Wikipedia) exhibits continuous entropy reduction across long sequences, whereas constrained literary forms such as poetry hit an early entropy plateau.



---

## Key Metrics & Mathematical Formulations

### Code Length $L(N)$

$$L(K) = \left\langle -\log_2 P_K(t_{K+1} \mid t_{1\dots K}) \right\rangle_{\text{data}}$$

* Measures the average number of bits required to compress and encode the actual target token $t_{K+1}$ given a sequence history of $K$ tokens.


* Maps token context length $K$ to character length $N$ using the empirical character-to-token ratio $N/K$ to ensure consistent cross-model comparisons.


* Serves as a strict empirical upper bound on the true conditional entropy of the underlying text distribution.



### Conditional Entropy $s(N)$

$$\mathcal{S}_{\text{cond}}(t_{1\dots K}) = -\sum_{t} P_K(t \mid t_{1\dots K}) \log_2 P_K(t \mid t_{1\dots K})$$

$$s(K) = \left\langle \mathcal{S}_{\text{cond}}(t_{1\dots K}) \right\rangle_{\text{data}}$$

* Quantifies the model's total intrinsic uncertainty across the full vocabulary space prior to observing the true target token.


* Tracks how broadly or narrowly the model spreads its probability distribution, complementing $L(K)$ which evaluates target-specific prediction accuracy.



### Surplexity $S_M(d)$

$$S_M(d) = \exp\left( -\frac{1}{m} \sum_{i=1}^{m} \ln q_i \right)$$

* Evaluates sequence-level perplexity as the exponentiated mean negative log-likelihood of target token probabilities $q_i = P(w_i \mid w_{0\dots i-1})$ over sequence length $m$.
* Low scores signify highly formulaic text; elevated scores indicate unexpected structural transitions.

### Token Surprise Heatmaps

$$\text{Surprise}_i = 1 - q_i$$

* Maps per-token surprise scores directly onto monospace text representations.
* Lightly shaded tokens indicate high predictability ($q_i \to 1$), while deep red highlights high-surprise transitions ($q_i \to 0$).

---

## Data Sources & Database Specifications

| Domain | Source / Endpoint | Precise Database & Technical Description |
| --- | --- | --- |
| **Simple Wiki** | Wikipedia REST API (`simple.wikipedia.org/api/rest_v1`) | Summary endpoints from Simple English Wikipedia. Text is strictly engineered for readability (Flesch-Kincaid Grade 6–8 level), characterized by controlled baseline vocabulary, simplified syntactic structures, low lexical variance, and explicit declarative phrasing. |
| **Finnegans Wake** | Public Domain Text Dump (Internet Archive) | Unabridged digital text of James Joyce's 1939 experimental narrative. Composes a high-entropy baseline characterized by extensive multilingual portmanteaus, nonce words, non-standard orthography, and subverted syntactic rules. |
| **Open Legal** | Federal Register REST API (`federalregister.gov/api/v1`) | Published rules, administrative notices, and executive orders from U.S. federal agencies. Contains high degrees of boilerplate legalese, rigid domain-specific phrasing, statutory cross-referencing, and long-range structural dependencies. |
| **Nigerian Pidgin** | `masakhane/masakhanews` (Subset: `pcm`) | Topic-classified news articles curated by the Masakhane NLP initiative for low-resource language benchmarks. Features an English-derived lexicon operating over West African substrate grammar, non-standard spelling conventions, and distinct morphosyntactic patterns. |
| **Synthetic (AI)** | `HuggingFaceTB/cosmopedia-v2` | Large-scale open synthetic dataset (25B+ tokens) generated primarily using Mixtral-8x7B across synthetic textbooks, course materials, and web rewrites. Exhibits low conditional entropy, high predictability, and narrow probability distributions due to model alignment and generation artifacts. |
| **Python Code** | `codeparrot/codeparrot-clean-valid` | De-duplicated, syntax-validated Python files scraped from public GitHub repositories. Characterized by strict Abstract Syntax Tree (AST) syntax rules, mandatory indentation formatting, deterministic keywords, and repeated variable scope patterns. |
| **Random Tokens** | Uniform Vocabulary Sampling ($t_i \sim U(0, \lvert V \rvert - 1)$) | Synthetic maximum-entropy baseline generated by uniformly sampling token IDs from the model's vocabulary space $V$. Establishes the empirical theoretical limit where token probability $P(t_i) = 1/\lvert V \rvert$ and token surprise is maximized. |

---

## Requirements & Setup

```bash
pip install torch transformers datasets matplotlib numpy requests

```

---

## Code Architecture

```
├── main.py
    ├── Setup: Loads Transformer backbone (AutoModelForCausalLM)
    ├── Data Fetchers: Streams data across 7 diverse text domains
    ├── Token Statistics Engine:
    │   ├── get_token_stats(): Computes NLL cross-entropy loss & surplexity
    │   └── get_code_length_curve_optimized(): Log-log context scaling (L(N) & s(N))
    ├── Visualizer Engine:
    │   ├── plot_heatmap(): Monospace token wrapping + color-mapped surprise
    │   └── plot_code_length_analysis(): Log-log plot of L(N) & s(N) vs N
    └── Execution Loop: Renders comparative heatmaps and scaling curves

```

---

## Running the Pipeline

```bash
python main.py

```

### Generated Output Panels:

1. **Surprise Heatmaps:** Renders tokenized passages with background colors mapped dynamically to target token surprise ($1 - q_i$).
2. **Context Scaling Dashboards:** Log-log plots measuring character context $N$ against code length $L(N)$ and conditional entropy $s(N)$.



---

## Citations

```bibtex
@article{scheibner2026large,
  title={Large language models and the entropy of English},
  author={Scheibner, Colin and Smith, Lindsay M. and Bialek, William},
  journal={arXiv preprint arXiv:2512.24969v1},
  year={2026}
}

@article{gambetta2026learning,
  title={Learning by Surprise: Adaptive Mitigation of Model Collapse in Large Language Models},
  author={Gambetta, Daniele and Gezici, Gizem and Giannotti, Fosca and Pedreschi, Dino and Knott, Alistair and Pappalardo, Luca},
  journal={Journal of the ACM (JACM)},
  volume={37},
  number={4},
  article={111},
  year={2026}
}

```
