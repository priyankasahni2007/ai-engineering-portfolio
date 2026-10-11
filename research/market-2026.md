# AI / ML Engineering Hiring Market — October 2026

_Run: 2026-10-11 (Phase A(a)). Purpose: decide what Priyanka's portfolio should prove, based on what postings actually ask for. Not a summary of hype._

## How to read this
- **Source quality varies.** Real postings (Google, Sarvam, NVIDIA, TikTok, Workiva, Marvell, Rippling, Kensho, Snowflake) are the strongest evidence. Job-description analyses (365 Data Science, ~900 Glassdoor postings; AI Shipping Labs, 1,000+ JDs) come next. Vendor and course blogs are weakest. Each claim below is tagged **[posting]**, **[JD-analysis]**, **[report]** or **[blog]**.
- Percentages from different sources measure different things and **do not reconcile**. Use them for direction, not precision.
- Several primary job pages could not be fetched from this environment (DNS blocked). Their requirements come from search-engine snippets of those pages. Re-verify before quoting any of them to a recruiter.

---

## 1. Headline findings (what repeats across sources)

| # | Signal | Evidence | Strength |
|---|---|---|---|
| 1 | **Python is the baseline.** PyTorch is the default DL framework. | 71% of ~900 postings require Python [JD-analysis: 365 Data Science]. PyTorch named in Rippling, Sarvam, Marvell, TikTok, OPPO postings [posting]. | Very strong |
| 2 | **RAG is the most common GenAI-specific requirement.** | RAG in 35.9% of 1,000+ AI-engineer JDs, ahead of prompt engineering (29.1%) and general LLM knowledge (25.4%) [JD-analysis: AI Shipping Labs]. LinkedIn: LangChain, RAG and PyTorch are the top AI-engineer skills [report via ODSC]. | Very strong |
| 3 | **Evaluation is the scarce, differentiating skill.** Golden datasets, offline faithfulness/hallucination metrics, retrieval-quality measurement, LLM-as-judge. | Snowflake FDE posting wants people who "care deeply about evals and quality" [posting]. Sarvam Applied AI Engineer lists "Evals, Observability" [posting via aggregator]. "AI Evals Engineer" is emerging as its own role in India (₹10–35 LPA) [blog]. Hiring-manager posts say evals are the part candidates skip [blog: Gauntlet]. | Strong |
| 4 | **Agents and tool use** (LangGraph, function calling, MCP, ReAct). | Sarvam Agents role: LangGraph, MCP, RAG, Evals, OAuth [posting]. Kensho intern: "designing and iterating on agentic systems" [posting]. Bengaluru postings: LangGraph/CrewAI, function calling [report]. | Strong and rising |
| 5 | **Production engineering around models.** FastAPI-style services, Docker, CI/CD, cloud (AWS 32.9%, Azure 26%), monitoring, drift, cost/latency control. | [JD-analysis: 365 Data Science], Bengaluru postings [report], Snowflake FDE (guardrails, observability) [posting]. | Strong |
| 6 | **CS fundamentals still gate internships.** DSA, software design, and distributed systems/concurrency. ML/AI is listed as a *preferred* qualification. | Google SWE Intern Summer 2027 (Bengaluru/Hyderabad/Pune) [posting]. Indian startups ask for DSA and system design [report: prepnplaced]. | Very strong for internships |
| 7 | **LLM inference/systems knowledge is a premium niche.** vLLM/SGLang/TensorRT-LLM, quantization (PTQ/QAT) with *measured* quality regressions, KV cache, batching, speculative decoding. | NVIDIA Quantized Inference roles; vLLM MLE roles [posting]. Marvell intern: GPU computing/CUDA [posting]. | Strong but senior-skewed. An intern who can *measure* these tradeoffs stands out. |
| 8 | **Classic ML (ranking/recsys) is still a large intern market at big tech.** Needs SQL/Pandas/Spark, CTR prediction, learning-to-rank, A/B testing and causal inference. | TikTok Search/Recommendation MLE Intern 2027 [posting]. OPPO intern: A/B testing + causal inference, recsys/CTR projects preferred [posting]. Senior recsys roles: LTR, CF, experiment logging [posting]. | Strong (big-tech / consumer apps) |
| 9 | **Multimodal / speech / vision** appears mainly as a *bonus*, not a core requirement, except at speech/vision-first companies. | Sarvam MLE: speech/audio/multimodal "a bonus"; Sarvam has an ML Engineer, Vision opening [posting]. Tripleten top-11 includes multimodal AI [blog]. | Moderate |
| 10 | **Early-career hiring in India values public GitHub proof over pedigree.** | Gnani.ai and Glean early-career roles expect a personal GitHub project [report]. Sarvam early-career guidance says the same [blog]. Sarvam fresher onsite: build a Voice Activity Detector from scratch in 2.5 h [blog, single account]. | Moderate (consistent, but secondary sources) |

## 2. Role lanes (pick the lane, then prove it)

| Lane | What they actually do | Key asks | Fit for Priyanka |
|---|---|---|---|
| **AI Engineer / Applied AI / GenAI Engineer** | Ship LLM features: RAG, agents, evals, cost/latency | Python, LLM APIs, RAG, vector DB, evals, LangGraph/MCP, FastAPI, Docker, cloud | **Primary target.** Most postings, and the fastest-growing title: LinkedIn has AI engineer as the fastest-growing US title (+143% YoY) and #2 on India Jobs on the Rise [report]. |
| **ML Engineer (classic)** | Train and serve models: ranking, recsys, fraud, forecasting | PyTorch, SQL/Spark, feature pipelines, offline/online eval, A/B tests | **Strong secondary.** Covers most big-tech MLE intern postings (TikTok, Workiva, Amazon-style roles). |
| **Forward Deployed / Applied AI (labs, Snowflake, Sarvam)** | Build with customers end-to-end, from scoping to eval to rollout | All of the above + communication + evals | Aspirational. The portfolio should *look like* FDE work: problem → system → measured result. |
| **ML Systems / Inference** | Make models fast and cheap | vLLM, quantization, CUDA/Triton, profiling | One "internals" project gives strong interview depth. Don't specialize yet. |
| **Research (Student Researcher, MSR India)** | Papers, experiments | Strong math, publications | Not the focus. A clean eval-heavy project still helps. |

## 3. Specific skills/tools that repeat → why they matter → where the portfolio proves it

| Skill / tool | Why employers want it | Prove it in |
|---|---|---|
| Hybrid retrieval (BM25 + dense), reranking, chunking | Most RAG failures are retrieval failures | PaperTrail (#1) |
| Retrieval metrics: Recall@K, MRR, nDCG | Shows you *measure* retrieval instead of eyeballing it | PaperTrail eval harness |
| Faithfulness / hallucination evals, golden sets, LLM-as-judge (with its limits) | Evals are the scarce skill (finding 3) | PaperTrail, then the agent project |
| Latency and cost per query | Production constraint that interviewers probe | PaperTrail per-config table |
| Agents: tool calling, structured outputs, state, retries, HITL, tracing | Rising demand (finding 4) | Agent flagship (Phase A(c)) |
| MCP | Named in real postings (Sarvam), but only use it where a tool-server boundary makes sense | Agent flagship, only if it fits |
| FastAPI + Docker + GitHub Actions CI | Baseline production signal | Every flagship |
| PyTorch training loop, fine-tuning | Required for most MLE roles | CV and/or ranking flagship |
| SQL/Pandas, feature pipelines, offline→online metrics, A/B design | Recsys/ranking MLE internships | End-to-end ML flagship (ranking/recsys) |
| Quantization / KV cache / inference benchmarking | Premium niche, interview depth | LLM-internals flagship |
| DSA + system design | Gates Google-style internships | Not a project. Note it as a parallel prep track in the roadmap. |

## 4. What does NOT carry signal (skip or de-emphasise)
- Prompt-only "chatbot over a PDF", or a Streamlit wrapper around one API call. Hiring-manager sources say one deep, evaluated project beats several generic demos [blog: dataexpert.io, myengineeringpath].
- Framework name-dropping without measurements. Postings ask for *measuring* quality regressions and retrieval quality, not for knowing LangChain exists.
- Certifications/"AI productivity tools" (from LinkedIn Skills on the Rise). They are relevant to generalists, not to engineering screens.
- Cybersecurity positioning. Out of scope for Priyanka by decision.

## 5. Implications for the portfolio (input to Phase A(b)/(c))
1. **PaperTrail is the right Project #1.** It sits on the two strongest signals (RAG + evals). Its differentiator must be the **evaluation harness and the per-config results table** (Recall@K/MRR/nDCG, faithfulness, p50/p95 latency, cost/query), not the chat UI.
2. The **agent flagship** should reuse PaperTrail as a tool, with real eval tasks and tracing. That gives a coherent story from retrieval to agent.
3. Include **one classic ML ranking/recsys project** with offline metrics and a simulated A/B/interleaving analysis. This covers the big-tech MLE intern lane (TikTok, OPPO-style).
4. Include **one LLM-internals project** focused on *measured* tradeoffs (e.g. quantization × KV-cache × batch size → tokens/s and quality delta on CPU). This is cheap to run and has high interview value.
5. Multimodal and CV are **bonus** signals. Choose ones with real evaluation and pick tasks that pair with Indian-market companies (speech/vision at Sarvam-type startups).
6. Every README must lead with **problem → measured result in one line**, because recruiters scan quickly [blog].
7. Run a **parallel DSA track** (not a portfolio item, but a gating factor for Google-type internships). Flag it in the roadmap.

## Sources
**Postings (primary or snippets of primary pages)**
- Google — Software Engineering Intern, Summer 2027 (India): https://google.com/about/careers/applications/jobs/results/120997883141857990-software-engineering-intern · https://jobs.anitab.org/companies/google-24698/jobs/82819261-software-engineering-intern-summer-2027
- Workiva — Summer 2027 Intern, MLE: https://workiva.wd503.myworkdayjobs.com/en-US/careers/job/Summer-2027-Intern---Machine-Learning-Engineering_R12194-1 · https://remotive.com/remote/jobs/artificial-intelligence/summer-2027-intern-machine-learning-engineering-5674971
- Rippling — ML Software Engineer Intern, Summer 2027: https://ats.rippling.com/rippling/jobs/f9de04c3-e2c5-41f9-8e46-a0684707f89d
- TikTok — MLE Intern (Data-Search-Recommendation), 2027: https://career.virginia.edu/jobs/tiktok-inc-machine-learning-engineer-intern-tiktok-data-search-recommendation-2027-summer/ · https://www.dice.com/job-detail/6def096d-8566-475f-bf93-a0225c1ee4b8
- Marvell — MLE Intern BS/MS, Summer 2027: https://marvell.wd1.myworkdayjobs.com/MarvellCareers/job/Santa-Clara-CA/Machine-Learning-Engineer-Intern--BS-MS---Summer-2027_2604989-1
- Kensho — MLE Summer Intern 2027: https://workopia.io/jobs/4a43beeafae3bcb2bdfd797477ae299e
- OPPO US Research — MLE Intern (recsys, A/B testing): https://ai-software-jobs.jobboardly.com/jobs/2025-machine-learning-engineer-intern-41710f3c
- Sarvam — ML Engineer: https://www.sarvam.ai/careers/jobs/25599 · careers: https://www.sarvam.ai/careers · Applied AI Engineer (Agents): https://www.metaintro.com/job/applied-ai-engineer-at-sarvam-d7bb8a15-92d5-5a49-a488-fa66553230d5
- NVIDIA — Senior MLE, Quantized Inference: https://jobs.anitab.org/companies/nvidia/jobs/68634387-senior-machine-learning-engineer-quantized-inference · Senior SWE, Quantized Inference: https://jobs.anitab.org/companies/nvidia/jobs/87315473-senior-software-engineer-quantized-inference
- Senior MLE, vLLM: https://platform-v2.wearedevelopers.com/en/jobs/ext/6034954/senior-machine-learning-engineer-vllm
- Anthropic FDE (aggregator): https://www.choppingblock.ai/jobs/forward-deployed-engineer-at-anthropic-49c92 · Forward Deployed AI Engineer: https://builtin.com/job/forward-deployed-ai-engineer/2722211
- Senior MLE, Recommendation Systems: https://jobs.digitalhire.com/job-listing/opening/1M6GNuiBnxLkvdv8maz2lX

**JD analyses and reports**
- AI Shipping Labs — What is an AI Engineer (1,000+ JDs): https://aishippinglabs.com/blog/what-is-an-ai-engineer-based-on-job-descriptions
- 365 Data Science — AI Engineer Job Outlook (~900 Glassdoor postings): https://365datascience.com/career-advice/career-guides/ai-engineer-job-outlook-2025/
- ODSC — 12 in-demand AI roles 2026 (cites LinkedIn Jobs on the Rise): https://opendatascience.com/12-in-demand-ai-job-roles-2026-how-to-get-hired/
- RecruitsLab — AI Engineer Hiring Report 2026: https://recruitslab.com/reports/ai-engineer-hiring-report-2026
- LinkedIn India coverage: https://www.businesstoday.in/india-at-100/story/ai-jobs-rise-51-in-india-but-75-of-job-skills-could-change-by-2030-linkedin-india-head-550758-2026-08-22 · https://www.storyboard18.com/brand-makers/linkedins-skills-on-the-rise-2026-identifies-top-skill-stacks-reshaping-indias-workforce-90556.htm · https://letsdatascience.com/news/bengaluru-anchors-india-ai-hiring-as-cities-expand-opportuni-37af2ef1
- Bengaluru market: https://www.360digitaltransformation.com/blogs/guides/ai-engineer-jobs-in-bangalore-2026-salary-skills-and-companies-hiring · https://www.weekday.works/jobs/ml-ai-engineering-jobs-in-bangalore
- India early-career lists: https://www.prepnplaced.com/apprenticeships/ai-and-ml · https://cloudsoftsol.com/blog/glean-ai-engineer-genai-fresher-bangalore-2026 · https://cloudsoftsol.com/blog/gnani-ai-engineer-voice-genai-fresher-bangalore-2026 · https://cloudsoftsol.com/blog/sarvam-ai-llm-engineer-fresher-early-career-bangalore-2026/

**Blogs (weak evidence, advice only)**
- https://dataskew.io/roadmaps/ai-engineering/ · https://gauntletai.com/night-school/what-hiring-managers-actually-look-for-in-ai-engineers · https://www.dataexpert.io/blog/ultimate-guide-ai-engineering-portfolios · https://myengineeringpath.dev/genai-engineer/portfolio-guide/ · https://futurense.com/blog/best-ai-engineering-roles · https://tripleten.com/blog/posts/ai-skills · https://myinternships.in/hire-ai-engineer-intern
