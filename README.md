# Statistical Inference Agent

An agentic engineering project that performs statistical inference on user-provided datasets — descriptive analysis, hypothesis testing, A/B testing, and linear/logistic regression — with a hybrid tool/declarative-operation architecture designed for auditability and safety.

## Status
🚧 In progress — Phases 1-8 complete, plus a dedicated stress-testing round (3 real bugs found and fixed). Next: product layer (simple tier) or medium-tier capabilities.

## Core design principle
The model chooses and interprets; deterministic code computes. The LLM never performs arithmetic or estimates a statistic itself — every number in a report comes from a tool call to real code (pandas / scipy / statsmodels), never from the model's own generation.

## Design decisions
| Area | Decision |
|---|---|
| Users | Analysts, stakeholders, and portfolio reviewers — one engine, two interaction modes |
| v1 ceiling | Hard tier (messy multi-file driver analysis), built in vertical slices: simple → medium → hard |
| Methods in scope | Descriptives, two/multi-group tests, A/B testing, linear regression, logistic regression |
| Methods out of scope (v1) | Survival analysis, prediction/ML, observational causal methods, count/Bayesian models |
| Core object | A structured analysis plan: draft → refined → frozen → deviations logged |
| Data access | Hybrid — typed tools for inference, declarative reviewable operations for wrangling |
| Data model | Handles, not rows; immutable versions with lineage; every operation logged with impact |
| Model visibility | Middle privacy tier by default — schema, types, value frequencies, format patterns; no raw rows |
| Evaluation | Synthetic data with planted ground truth + public/real data for process and robustness |
| Orchestration | LangGraph — explicit stages, plan-freeze checkpoints, approval interrupts |
| Decision auditability | Every method selection and wrangling choice records a rationale, alternatives considered, and evidence — later changes link back (`supersedes`) to what they replaced and why |

## Tech stack
- **Orchestration:** LangChain + LangGraph
- **Model:** Groq (`openai/gpt-oss-120b`) via `langchain-groq` — chosen after Gemini's free tier proved too rate-limited during the companion Text-to-SQL project; verified with real token-budget testing (see commit history) rather than trusting aggregator pricing pages
- **Statistical computation:** scipy, statsmodels, pandas — never the model itself
- **Schemas:** pydantic

## A note on model choice
Groq's production model catalog changed during this build — Llama models (originally the planned choice) were deprecated from Groq's platform in favor of OpenAI's open-weight GPT-OSS models. This project uses `openai/gpt-oss-120b`, Groq's higher-capability option, given this agent's error mode (a confident wrong statistical conclusion) is more consequential than the Text-to-SQL agent's (a wrong retrieved value).

## Setup

### 1. Clone and create a virtual environment
\`\`\`bash
git clone https://github.com/suhaspadi-work/stats-inference-agent.git
cd stats-inference-agent
python3 -m venv .venv
source .venv/bin/activate
\`\`\`

### 2. Install dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. Set up environment variables
Create a `.env` file in the project root with:
\`\`\`
GROQ_API_KEY=your_groq_api_key_here
\`\`\`
Get a free Groq API key at https://console.groq.com/keys.

### 4. Verify setup
\`\`\`bash
python test_connection.py
\`\`\`
Should print a response from the model, confirming your API key and dependencies are working.

## Known limitations
- **LangGraph checkpoint serialization coerces tuples to lists** across a pause/resume (interrupt) boundary, since the underlying format (JSON/msgpack) has no native tuple type. `AnalysisPlan`'s append methods are defensively written to tolerate either form.
- **Values computed via numpy/scipy must be explicitly cast to plain Python types** before being placed into graph state — numpy scalar types (e.g. `numpy.float64`) are not serializable by LangGraph's checkpointer and will fail only when a checkpoint boundary is actually crossed, not on every run.
- **Custom dataclass/Enum types used in graph state are not yet registered with LangGraph's serializer**, producing deprecation warnings on every checkpoint read. Works today; will need explicit registration in a future LangGraph version.
- **Auto-approved wrangling operations (stakeholder mode) are not distinguished from human-approved ones in the Operation audit record itself** — both produce an identical APPROVED Operation. Full attribution of "who/what approved this" would require an additional field on Operation; left as a known gap rather than a silent one.

## Evaluation results (Phase 8)

A 26-scenario battery run against the full agent end-to-end (not individual components): 20 A/A trials (no real effect), 4 planted-effect trials of varying magnitude, and 2 messy-data trials (duplicates/missingness, auto-cleaned).

| Category | Result |
|---|---|
| A/A false-positive rate | 1/20 = 5.0% (theoretical expectation: 5%) |
| Effect-detection scenarios | 4/4 passed — planted effects correctly recovered across small/medium/large magnitudes |
| Messy-data scenarios | 2/2 passed — correct statistical conclusions after real duplicates/missingness were detected and cleaned |
| **Overall** | **26/26 passed** |

Raw results: `eval/phase8_results.jsonl`. Run `python -m eval.run_all` to reproduce (resumable; takes 15-25 minutes given rate-limit pacing), then `python eval/summarize.py` for the aggregate report.

One real bug was found and fixed via this evaluation run: `execute_filter` crashed when a `not_null`/`is_null` condition's params correctly omitted the irrelevant `value` key — a gap in the original Phase 4 implementation, only exposed once the model began generating its own filter proposals with realistically-shaped (and correctly minimal) parameters.

## Stress testing and edge cases

Beyond Phase 8's statistical battery, a separate round of targeted stress testing probed structural failure modes, reasoning edge cases, and product-relevant multi-step scenarios. Three real bugs were found and fixed:

1. **Non-numeric outcome column** — previously reached `scipy` uncaught, crashing with a confusing `TypeError`. Now validated at plan-drafting time with a clear, early `ValueError`.
2. **Invalid predictor edit at freeze** — editing the plan's predictors at the human-approval step to a nonexistent column previously crashed deep in execution with a bare `KeyError`. Now validated against the real dataset at the edit point itself.
3. **Privacy leak in high-cardinality columns** — `top_values` in `ModelSafeView` was populated unconditionally regardless of cardinality, meaning a column like raw email addresses (200 distinct values) leaked real values through to the model. Fixed with a cardinality gate; the existing test whose name claimed to cover this case was strengthened, since it had never actually checked `top_values` at all.

Also verified, with no issues found:
- Chained rejections (two data-quality issues rejected in one wrangling checklist — both tracked and surfaced correctly)
- Stakeholder-mode autonomy correctly always interrupts on missingness, never auto-approving it
- Classification stability across repeated identical requests
- Performance at 5,000 rows (no degradation)
- Unicode/non-ASCII values throughout the pipeline
- Multi-turn sessions: a second question against the same dataset correctly starts a fresh plan and builds on the first question's cleaned data lineage, not the raw original
- Multi-interrupt sequencing (wrangling checklist followed by plan approval)

**Known gaps, not yet tested:** the specific combination of an ambiguous *and* simultaneously messy dataset triggering classify's clarification gate together with wrangling's checklist in the same run; adversarial (as opposed to merely awkward) inputs designed to defeat guards; partial failure mid-sequence (e.g., a service outage after some wrangling operations already executed).

## Roadmap
- [x] Environment + git setup
- [x] Core dependencies installed
- [x] Model connectivity verified (Groq / gpt-oss-120b via langchain-groq)
- [x] Core data structures (Dataset, Operation, AnalysisPlan schemas, ModelSafeView privacy boundary, EventLog persistence) — 20 automated pytest tests, all passing
- [x] Ingestion and profiling tools (load_dataset, profile_dataset — 30 automated tests, all passing)
- [x] Synthetic data generator with planted ground truth (A/A and planted-effect scenarios — 37 automated tests total, all passing)
- [x] Simple-tier wrangling operations (cast, rename, filter, dedupe — approval-gated, rationale-tracked, 48 automated tests total, all passing)
- [x] Hypothesis testing tools (two-group test with assumption-driven selection between t-test/Welch/Mann-Whitney, verified against direct scipy calls — 53 automated tests total, all passing)
- [x] Plan classifier and simple-tier agent loop (full LangGraph agent: classify → profile → draft_plan → select_method → freeze → execute → report, with both interrupt gates working — 88 automated tests, all passing, including one full live end-to-end run)
- [x] Human-in-the-loop and autonomy dial (plan-freeze approval, wrangling checklist interrupt, clarification gate, and a configurable analyst/stakeholder autonomy mode — 121 automated tests, all passing, including one full end-to-end proof with genuinely messy data and a deliberate rejection)
- [x] Simple-tier evaluation (26-scenario battery: 20 A/A trials, 4 effect-size scenarios, 2 messy-data scenarios — 5.0% A/A false-positive rate, 26/26 overall pass rate)
- [ ] Medium-tier: EDA and multi-group tools
- [ ] Linear regression tools
- [ ] Multi-file data assembly (hard tier)
- [ ] Feature construction and logistic regression (hard tier)
- [ ] Full hard-tier integration
- [ ] Reporting layer and interaction modes
- [ ] Real-world evaluation
- [ ] Documentation and portfolio write-up

See the full phase-wise blueprint (build reference document) for detailed checkpoints per phase.