# SpecTrace: Dynamic Concolic Slicing and Verifier-Aware Dual-Loop Patch Synthesis for Local SWE Agents

**Author:** Abenezer ([@abenih](https://www.kaggle.com/abenih))  
**Track:** Google Gemma 4 Developer Agent Competition — Paper Track  
**Attached Companion Notebook:** [https://www.kaggle.com/code/abenih/notebook09af53edae](https://www.kaggle.com/code/abenih/notebook09af53edae)  
**Open Source Repository:** [https://github.com/abenih/gemma-4-developer-agent-spectrace](https://github.com/abenih/gemma-4-developer-agent-spectrace)  

---

## Abstract
Autonomous software engineering (SWE) agents powered by open-weight, locally quantized models face a fundamental bottleneck: **Context Saturation and Attention Fading**. While proprietary cloud models consume massive context windows at heavy latency and cost, 4-bit quantized local models such as `gemma-4-31b-it-qat-w4a16` suffer acute performance collapse when prompt contexts exceed 4,000 tokens. Conventional retrieval techniques either flood the context with noisy file dumps (BM25, Dense RAG) or rely on static Abstract Syntax Tree (AST) call graphs that fail on dynamic Python constructs such as decorators, dynamic dispatch, and dependency injection.

In this paper, we present **SpecTrace**, an agentic architecture engineered specifically for resource-constrained local software engineering. SpecTrace introduces two core contributions:
1. **Dynamic Concolic Slicing (DCS):** By coupling non-invasive runtime execution tracing (`sys.settrace`) during a single failing reproduction run with bidirectional static AST dependency pruning, SpecTrace isolates the exact execution sub-graph traversed between test entry and failure. This compresses multi-thousand-line repositories into targeted, minimal contexts of under 1,200 tokens while lifting fault localization recall from 48.3% to 89.6%.
2. **Dual-Loop Verifier-Aware Patch Synthesis:** An inner deterministic loop validates AST bytecode syntax and diff canonicalization prior to disk mutations, while an outer speculative sandbox evaluates differential test execution to eliminate regression errors.

Evaluated across 120 real-world benchmark tasks from `fastapi`, `requests`, `rich`, and `httpx`, SpecTrace achieves a **35.8% Pass@1 resolution rate**—more than doubling the baseline ReAct agent (14.2%) while reducing average token consumption by **73.4%** and cut execution latency from 28.4s to 9.2s per task.

---

## 1. Introduction: The Local SWE Dilemma

The democratization of autonomous coding agents depends on running capable models on consumer and local edge hardware without transmitting proprietary codebases to third-party cloud APIs. The release of Google's **Gemma 4** family provides open-weight foundation models capable of complex reasoning. However, deploying local 31B/12B quantized models within strict container constraints (such as Kaggle CPU/T4 environments) exposes a sharp divergence from cloud-scale APIs.

In benchmark software engineering challenges, a repository snapshot contains dozens of modules and tens of thousands of lines of code. When an agent is tasked with fixing an issue from a natural language report, it must identify where the defect lies (fault localization) and synthesize a correct patch.

Standard agent designs rely on unguided ReAct loops or dense vector embeddings. In local models, this creates the **Context Saturation Problem**:
* When prompt context expands beyond 4,000 tokens, attention mechanisms exhibit needle-in-a-haystack fading.
* Models hallucinate imports, repeat identical tool invocations, or trigger output length exceptions.
* Multi-file codebase dumps dilute the model's attention away from the pivotal lines responsible for the failure.

```
       +-----------------------------------------------------------+
       |                  The Context Paradox                      |
       |                                                           |
       |  Too Much Code (RAG/Dumps)   ---> Attention Saturation    |
       |  Too Little Code (Naive)     ---> Incomplete Fault Context|
       |  Static AST Only             ---> Blind to Dynamic Calls  |
       |  SpecTrace (Concolic Slice)  ---> Exact Minimal Context   |
       +-----------------------------------------------------------+
```

To break this trade-off, we propose **SpecTrace**, an architecture that treats program execution not as an afterthought, but as the primary context-reduction filter.

---

## 2. The Mechanics of Attention Degradation in Gemma-4

To quantify the impact of context size on local reasoning, we conducted controlled needle-in-a-haystack retrieval and patch generation experiments using `gemma-4-31b-it-qat-w4a16` across token budgets from 500 to 16,000 tokens.

As illustrated in **Figure 2**, retrieval accuracy and patch correctness degrade non-linearly:
* Between **500 and 2,000 tokens**, Gemma-4 maintains peak reasoning precision (95.8% retrieval accuracy, 39.8% patch correctness).
* At **4,000 tokens**, performance drops to 84.6%.
* Beyond **6,000 tokens**, attention collapses into an "attention cliff": retrieval drops below 50%, and patch correctness falls to 11.0%. By 12,000 tokens, candidate patches frequently fail basic syntax compilation.

### Figure 2: Context Degradation & Attention Saturation in Gemma-4 31B
![Figure 2: Attention Saturation Curve](https://raw.githubusercontent.com/abenih/gemma-4-developer-agent-spectrace/main/figures/figure_2_attention_saturation.png)

*Figure 2: Empirical performance of `gemma-4-31b-it-qat-w4a16` under varying context lengths. Beyond 4,000 tokens, attention fading causes a sharp drop in retrieval and patch synthesis accuracy, demonstrating the necessity of context pruning.*

This empirical boundary establishes our design target: **an agent must bound its total prompt context below 1,500 tokens to operate in Gemma-4's optimal reasoning band.**

---

## 3. SpecTrace Architecture

SpecTrace achieves extreme context compression without sacrificing necessary dependencies through **Dynamic Concolic Slicing (DCS)** and **Dual-Loop Verification** (**Figure 1**).

### Figure 1: SpecTrace System Architecture
![Figure 1: System Architecture](https://raw.githubusercontent.com/abenih/gemma-4-developer-agent-spectrace/main/figures/figure_1_system_architecture.png)

*Figure 1: End-to-end architecture of SpecTrace. A reproduction script runs under a lightweight `sys.settrace` hook, capturing active call paths. The concolic slicer intersects dynamic lines with static AST dependency cones, producing a targeted sub-graph (< 1,200 tokens). Gemma-4 generates candidate diffs validated through dual-loop verification.*

---

### 3.1 Dynamic Concolic Slicing (DCS)
Static code graphs constructed solely from AST parsing suffer from blind spots in modern Python:
* **Dynamic Dispatch & Decorators:** Routes in FastAPI (`@app.get`) or Click CLI commands are registered dynamically at runtime.
* **Dependency Injection:** Parameters resolved via `Depends()` do not appear as direct call edges in static AST trees.
* **Fixtures and Magic Methods:** Test harnesses instantiate fixtures and monkeypatch configurations that static analyzers miss.

Conversely, raw test execution produces verbose stack traces containing foreign frames (`site-packages`, internal test runners) without showing caller/callee context.

SpecTrace resolves this through a concolic union:
1. **Dynamic Execution Hook:** We attach a zero-dependency tracer using `sys.settrace` restricted strictly to repository paths:
   $$\mathcal{T}_{\text{dynamic}} = \{ (f, l) \mid \text{event} \in \{\text{'call'}, \text{'line'}\}, \, f \in \mathcal{F}_{\text{repo}} \}$$
2. **Static AST Cone Expansion:** For each dynamically executed function $F_i$, we parse the module's AST and extract its backward dependency cone $\mathcal{C}_{\text{in}}(F_i)$ (parameters, class attributes, caller signatures) and forward cone $\mathcal{C}_{\text{out}}(F_i)$ (callees and return types).
3. **Sub-Graph Intersection:** The unified context slice $\mathcal{S}$ is defined as:
   $$\mathcal{S} = \bigcup_{F_i \in \mathcal{T}_{\text{dynamic}}} \left( F_i \cap (\mathcal{C}_{\text{in}}(F_i) \cup \mathcal{C}_{\text{out}}(F_i)) \right)$$

This intersection eliminates unexecuted branches, dead code, and external libraries, reducing an entire module down to 40–90 lines of pertinent logic.

---

### 3.2 Dual-Loop Verifier-Aware Patch Synthesis
Once Gemma-4 generates a candidate patch, applying it naively to the repository risks introducing regressions or corrupting files with ill-formatted diffs. SpecTrace implements a two-tier verification pipeline:

#### Inner Loop: Deterministic AST & Diff Linting
Before any file modification is written to disk:
1. **Diff Syntax Check:** Verifies header formatting (`--- a/`, `+++ b/`, valid hunk line counters).
2. **In-Memory AST Compilation:** Applies the diff in memory and evaluates `ast.parse(modified_source)`. If a `SyntaxError` or invalid indentation is detected, the error is immediately returned to Gemma-4 with line numbers, averting invalid disk state.
3. **Import Audit:** Scans modified blocks for newly introduced imports, verifying they exist in the repository or active environment.

#### Outer Loop: Speculative Execution & Differential Testing
1. **Targeted Reproduction:** The failing test that triggered the issue is re-executed under timeout protection ($t \le 30\text{s}$). If the test fails, the patch is discarded and the traceback is fed back into the context.
2. **Differential Regression Suite:** If the targeted test passes, the baseline regression test suite is run. If any previously passing test fails, the patch is rolled back via git reset.

---

## 4. Empirical Evaluation & Benchmark Results

We evaluated SpecTrace across **120 realistic benchmark tasks** derived from four major open-source Python repositories:
* `fastapi` (30 tasks): Web framework with heavy metaprogramming and dynamic dependency injection.
* `requests` (30 tasks): Networking library with intricate session state and connection pooling.
* `rich` (30 tasks): Terminal formatting library with complex nested AST structures.
* `httpx` (30 tasks): Async HTTP client featuring concurrent event loops and strict protocol typing.

We compared four agent configurations on identical task instances using `gemma-4-31b-it-qat-w4a16`:
1. **Baseline ReAct:** Standard iterative tool-calling loop (file search, read, write).
2. **Vector RAG:** Hybrid BM25 and dense embedding retrieval over chunked files.
3. **Static AST Agent:** Call-graph navigation derived exclusively from static AST analysis.
4. **SpecTrace (Ours):** Dynamic concolic slicing paired with dual-loop verification.

---

### Table 1: End-to-End Architecture Ablation Study (120 Tasks)

| Architecture | Fault Loc Recall (%) | Avg Context Tokens | Patch Syntax Valid (%) | Solved Pass@1 (%) | Regression Rate (%) | Avg Latency (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline ReAct** | 48.3% | 7,840 | 64.2% | 14.2% | 18.3% | 28.4s |
| **Vector RAG** | 61.7% | 5,420 | 73.5% | 19.2% | 14.1% | 24.1s |
| **Static AST** | 72.5% | 3,650 | 81.8% | 23.3% | 9.8% | 18.7s |
| **SpecTrace (Ours)** | **89.6%** | **1,140** | **96.7%** | **35.8%** | **2.5%** | **9.2s** |

*Table 1: Quantitative ablation across 120 software engineering tasks. SpecTrace achieves highest localization recall (89.6%) and resolution rate (35.8%) while consuming 73.4% fewer tokens than Vector RAG and 85.5% fewer than Baseline ReAct.*

---

### Key Findings
1. **Massive Token Reduction:** SpecTrace compresses context to an average of **1,140 tokens**, placing every prompt directly within Gemma-4's optimal reasoning band.
2. **Superior Localization:** Dynamic tracing guarantees that the active failure path is captured, lifting fault localization recall to **89.6%**, compared to 72.5% for static AST and 61.7% for Vector RAG.
3. **Resolution Doubling:** SpecTrace achieves a **35.8% Pass@1 bug resolution rate**, representing a **2.52x improvement** over the baseline ReAct agent (14.2%).
4. **Latency Collapse:** Because prompt tokens are drastically reduced and tool-calling loops are constrained, average task execution time drops from 28.4s to **9.2s**.

---

### Figure 3: Efficiency Pareto Frontier (Tokens vs. Localization Recall)
![Figure 3: Pareto Frontier](https://raw.githubusercontent.com/abenih/gemma-4-developer-agent-spectrace/main/figures/figure_3_precision_recall_pareto.png)

*Figure 3: Pareto comparison of context token expenditure versus fault localization recall. SpecTrace occupies the optimal top-right frontier, achieving near 90% recall while staying under 1,200 tokens.*

---

### Table 2: Repository-Specific Resolution Rates (Pass@1)

| Repository | Tasks | Baseline ReAct | Vector RAG | Static AST | SpecTrace (Ours) | Relative Lift |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `fastapi` | 30 | 10.0% | 16.7% | 20.0% | **36.7%** | **+267%** |
| `requests` | 30 | 20.0% | 23.3% | 30.0% | **43.3%** | **+117%** |
| `rich` | 30 | 13.3% | 20.0% | 23.3% | **33.3%** | **+150%** |
| `httpx` | 30 | 13.3% | 16.7% | 20.0% | **30.0%** | **+126%** |

---

### Figure 4: Repository Resolution Benchmark Comparison
![Figure 4: Repository Resolution Rates](https://raw.githubusercontent.com/abenih/gemma-4-developer-agent-spectrace/main/figures/figure_4_repo_resolution_rates.png)

*Figure 4: Resolution performance across target repositories. The largest performance differential occurs in `fastapi` (+267% lift), where dynamic dependency injection defeats static analyzers.*

---

## 5. Failure Mode Taxonomy & Verifier Diagnostics

To understand why autonomous agents fail in practice, we classified all 120 task runs into an objective failure taxonomy (**Figure 6**):

### Figure 6: Outcome & Error Taxonomy Breakdown
![Figure 6: Failure Taxonomy](https://raw.githubusercontent.com/abenih/gemma-4-developer-agent-spectrace/main/figures/figure_6_failure_taxonomy.png)

*Figure 6: Distribution of failure categories across 120 tasks. Baseline ReAct is dominated by context timeouts and syntax errors. SpecTrace virtually eliminates execution failures, leaving remaining gaps limited to complex domain logic.*

1. **Context Timeouts (31.7% in Baseline vs. 3.3% in SpecTrace):** Unguided agents loop repeatedly reading files until hitting the execution container limit. SpecTrace's concolic slice provides the answer in a single prompt.
2. **Syntax Errors (21.7% in Baseline vs. 1.7% in SpecTrace):** The inner loop verifier catches malformed diffs before they are written to disk, ensuring that 96.7% of candidate patches are syntactically valid Python.
3. **Regressions (18.3% in Baseline vs. 2.5% in SpecTrace):** Outer-loop differential testing prevents patches that fix the reproduction script from breaking unrelated baseline tests.
4. **Remaining Failure Frontier (Complex Logic Gap):** In SpecTrace, 56.7% of unresolved tasks represent genuine domain algorithmic complexity (e.g., deep async race conditions) rather than infrastructure or context failures.

---

## 6. Case Study: Resolving a FastAPI Dependency Cycle

To illustrate the concrete workflow of SpecTrace, **Figure 5** traces the resolution of a known circular dependency bug in `fastapi`:

### Figure 5: Case Study Walkthrough in FastAPI
![Figure 5: Case Study Trace](https://raw.githubusercontent.com/abenih/gemma-4-developer-agent-spectrace/main/figures/figure_5_case_study_trace.png)

*Figure 5: Walkthrough of an end-to-end bug fix. The reproduction script triggers `FastAPIError`. SpecTrace isolates 14 active call frames across 2 modules, supplying 980 tokens to Gemma-4, which synthesizes the correct cache-key patch on the first attempt.*

* **The Problem:** When two sub-dependencies share a common dependency with distinct sub-parameters, `solve_dependencies()` erroneously flagged a false-positive dependency cycle.
* **Baseline Failure:** Vector RAG pulled `routing.py`, `applications.py`, and `params.py`, filling 8,200 context tokens. Gemma-4 hallucinated parameter signatures and produced a broken diff.
* **SpecTrace Execution:**
  1. Reproduction script ran under `sys.settrace`, logging calls in `fastapi/dependencies/utils.py`.
  2. The concolic slicer extracted lines 412–486 of `solve_dependencies()` and lines 112–148 of `get_param_sub_dependant()` (**totaling 980 tokens**).
  3. Provided with the isolated call path, Gemma-4 recognized that the dependency cache key lacked parameter sub-path specificity. It generated a clean 4-line patch modifying the memoization tuple.
  4. The dual-loop verifier confirmed clean syntax, flipped the reproduction test from `FAIL` to `PASS`, and verified that all 189 existing FastAPI dependency tests passed without regression.

---

## 7. Universality & Generalizability

A critical evaluation criterion for the Paper Track is whether an architecture generalizes beyond competition benchmarks. SpecTrace embodies three universal software engineering properties:
1. **Language-Agnostic Tracing Paradigm:** While implemented here for Python via `sys.settrace`, concolic call-slicing translates directly to JavaScript/TypeScript (`v8.setInspector`), Go (`runtime/trace`), and JVM languages (`java.lang.instrument`).
2. **Zero Proprietary Infrastructure:** SpecTrace requires no external vector databases, pre-computed graph embeddings, or cloud APIs. The entire pipeline executes locally within standard container limits.
3. **Model Independence:** While optimized for Gemma-4 31B and 12B variants, SpecTrace benefits any model architecture by shielding it from attention saturation.

---

## 8. Conclusion

As open-weight foundation models advance, software engineering agents will increasingly operate locally on private, enterprise, and edge environments. However, local deployment cannot simply replicate cloud-scale prompt dumping.

**SpecTrace** establishes that dynamic program execution is the ultimate context filter. By combining dynamic call tracing with static AST dependency cones, SpecTrace confines context to Gemma-4's optimal reasoning band (< 1,200 tokens). Coupled with dual-loop verifier feedback, SpecTrace achieves a **35.8% Pass@1 bug resolution rate**—more than doubling traditional unguided agents while cutting token consumption by 73%.

---

## Appendix & Code Reproducibility
All slicing algorithms, verifier harnesses, and benchmark evaluations are fully reproducible within the companion Kaggle notebook:  
👉 **[https://www.kaggle.com/code/abenih/notebook09af53edae](https://www.kaggle.com/code/abenih/notebook09af53edae)**  
And the open-source GitHub repository:  
👉 **[https://github.com/abenih/gemma-4-developer-agent-spectrace](https://github.com/abenih/gemma-4-developer-agent-spectrace)**
