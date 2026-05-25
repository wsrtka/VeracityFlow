# VeracityFlow

A portfolio project for automated fact-checking using agentic AI workflows. VeracityFlow takes a claim, researches it using live web search, then runs a structured multi-agent debate to produce a final veracity verdict with a confidence score.

## How It Works

The pipeline is built on three frameworks working in concert:

**LangGraph** orchestrates the overall workflow as a stateful graph. A claim enters the graph and passes through a sequence of nodes — research planning, web search, and agent debate — before producing a final report.

**DSPy** handles the LLM-powered reasoning steps. A `ChainOfThought` module generates targeted search queries from the input claim, using Gemini 2.5 Flash as the underlying model via the Gemini AI Studio API.

**AutoGen (autogen-agentchat)** runs the multi-agent debate at the end of the pipeline. A `FactChecker` and a `Contextualist` agent argue the claim against the retrieved evidence, and a `Judge` agent reviews the debate and issues a final Veracity Score (0–100) with a summary.

Web search is powered by the **Tavily** search API with advanced depth and raw content retrieval.

```
claim input
    │
    ▼
[LangGraph]
    │
    ├── research_planner (DSPy ChainOfThought → search queries)
    │
    ├── web_search (Tavily API → evidence)
    │
    └── agent_debate (AutoGen: FactChecker ↔ Contextualist → Judge)
                │
                ▼
         final report + veracity score
```

## Stack

| Component | Library |
|-----------|---------|
| Workflow orchestration | `langgraph` |
| LLM reasoning | `dspy` |
| Multi-agent debate | `autogen-agentchat`, `autogen-ext[openai]` |
| Web search | `tavily-python` |
| Language model | Gemini 2.5 Flash (via Gemini AI Studio) |
| Containerisation | Docker / Docker Compose |

## Getting Started

### Prerequisites

- Docker and Docker Compose
- A [Gemini API key](https://aistudio.google.com/app/apikey) (free tier available)
- A [Tavily API key](https://tavily.com/)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/wsrtka/VeracityFlow.git
   cd VeracityFlow
   ```

2. Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

3. Build and run with Docker Compose:
   ```bash
   docker compose up --build
   ```

The app runs `app/fact_checker.py` on startup.

### Running Locally (without Docker)

Requires Python 3.13 (autogen-agentchat does not yet support Python 3.14).

```bash
pip install -r requirements.txt
python app/fact_checker.py
```

## Project Structure

```
VeracityFlow/
├── app/
│   └── fact_checker.py     # Main entrypoint and LangGraph pipeline
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env                    # Not committed — create locally (see Setup)
```

## Notes

- The Tavily free tier has rate limits. If you hit them during development, add a short delay between searches or reduce `max_results`.
- The `google-cloud-aiplatform` package is not required for this project. Authentication goes through the Gemini AI Studio API directly, not Vertex AI.
