import os
from typing import List, TypedDict

from langgraph.graph import END, StateGraph
from tavily import TavilyClient


class AgentState(TypedDict):
    claim: str
    queries: List[str]
    search_results: List[str]
    is_sufficient: bool
    final_report: str


def research_planner(state: AgentState):
    print("PLANNING RESEARCH")
    return {"queries": [f"Fact check: {state['claim']}", "Evidence for and against"]}


def search_engine(state: AgentState):
    print("SEARCHING")
    response = tavily_client.search(
        state["queries"][0],
        search_depth="advanced",
        max_results=3,
        include_raw_content=True,
    )
    new_results = [r["content"] for r in response["results"]]
    return {"search_results": state.get("search_results", []) + new_results}


def sufficiency_checker(state: AgentState):
    print("CHECKING SUFFICIENCY")
    if len(state.get("search_results", [])) > 3:
        return "sufficient"
    return "insufficient"


workflow = StateGraph(AgentState)

workflow.add_node("planner", research_planner)
workflow.add_node("searcher", search_engine)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "searcher")
workflow.add_conditional_edges(
    "searcher",
    sufficiency_checker,
    {
        "sufficient": END,
        "insufficient": "searcher",
    },
)

app = workflow.compile()


if __name__ == "__main__":
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    initial_input = {"claim": "The moon is made of cheese"}
    result = app.invoke(initial_input)
    print(result)
