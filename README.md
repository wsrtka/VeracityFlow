# VeracityFlow

A portfolio project for automated fact-checking using agentic AI workflows. VeracityFlow takes a claim, researches it using live web search, then runs a structured multi-agent debate to produce a final veracity verdict with a confidence score.

## How It Works

The pipeline is built on three frameworks working in concert:

**LangGraph** orchestrates the overall workflow as a stateful graph. A claim enters the graph and passes through a research planning node, a search node (which loops up to 3 times to gather sufficient evidence), and finally a reporting node that produces the final verdict.

**DSPy** handles the LLM-powered reasoning steps. A `ChainOfThought` module with a typed `GenerateSearchQueries` signature generates exactly 3 targeted search queries from the input claim, using Gemini 2.5 Flash as the underlying model via the Gemini AI Studio API.

**AutoGen (autogen-agentchat)** runs the multi-agent debate at the end of the pipeline. A `FactChecker` and a `Contextualist` agent debate the claim against the retrieved evidence for 4 messages, after which a `Judge` agent reviews the transcript and issues a final Veracity Score (0–100) with a summary.

Web search is powered by the **Tavily** search API with advanced depth and raw content retrieval.

```
claim input
    │
    ▼
[LangGraph]
    │
    ├── research_planner (DSPy ChainOfThought → 3 search queries)
    │
    ├── search_engine (Tavily API → evidence) ◄─┐
    │         │                                  │
    │   sufficiency_checker                      │
    │         ├── insufficient (revision < 3) ───┘
    │         └── sufficient
    │
    └── final_report_node
              │
              ├── debate: FactChecker ↔ Contextualist (4 messages)
              └── Judge reviews transcript → Veracity Score + summary
                        │
                        ▼
                  final_report
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

The app runs `app/fact_checker.py` on startup, which by default fact-checks the claim `"The moon is made of cheese."` — swap it out for your own claim at the bottom of the file.

### Running Locally (without Docker)

Requires Python 3.13 (`autogen-agentchat` does not yet support Python 3.14).

```bash
pip install -r requirements.txt
python app/fact_checker.py
```

## Project Structure

```
VeracityFlow/
├── app/
│   └── fact_checker.py     # Full pipeline: DSPy signatures, LangGraph graph, AutoGen debate
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env                    # Not committed — create locally (see Setup)
```

## Notes

- The Tavily free tier has rate limits. The search node runs up to 3 iterations, so a single run makes up to 3 API calls. If you hit rate limits during development, reduce the iteration cap in `sufficiency_checker` or add a `time.sleep()` between calls.
- Authentication to Gemini goes through the AI Studio API directly using `GEMINI_API_KEY` — the `google-cloud-aiplatform` package is not required and can be removed from `requirements.txt`.
