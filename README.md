# Statistical Inference Agent

An agentic engineering project that performs statistical inference on user-provided datasets — descriptive analysis, hypothesis testing, A/B testing, and linear/logistic regression — with a hybrid tool/declarative-operation architecture designed for auditability and safety.

## Status
🚧 In progress — currently at Phase 1 (core data structures) of the build.

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

## Roadmap
## Roadmap
- [x] Environment + git setup
- [x] Core dependencies installed
- [x] Model connectivity verified (Groq / gpt-oss-120b via langchain-groq)
- [x] Core data structures (Dataset, Operation, AnalysisPlan schemas, ModelSafeView privacy boundary, EventLog persistence) — 20 automated pytest tests, all passing
- [ ] Ingestion and profiling tools
- [ ] Synthetic data generator with planted ground truth
- [ ] Simple-tier wrangling operations
- [ ] Hypothesis testing tools
- [ ] Plan classifier and simple-tier agent loop
- [ ] Human-in-the-loop and autonomy dial
- [ ] Simple-tier evaluation
- [ ] Medium-tier: EDA and multi-group tools
- [ ] Linear regression tools
- [ ] Multi-file data assembly (hard tier)
- [ ] Feature construction and logistic regression (hard tier)
- [ ] Full hard-tier integration
- [ ] Reporting layer and interaction modes
- [ ] Real-world evaluation
- [ ] Documentation and portfolio write-up

See the full phase-wise blueprint (build reference document) for detailed checkpoints per phase.