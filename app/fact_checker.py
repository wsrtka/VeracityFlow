import asyncio
import os
from typing import List, TypedDict

import dspy
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from langgraph.graph import END, StateGraph
from tavily import TavilyClient

gemini_model = dspy.LM(
    model="gemini/gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"), max_tokens=500
)
dspy.settings.configure(lm=gemini_model)


class AgentState(TypedDict):
    claim: str
    queries: List[str]
    search_results: List[str]
    revision_number: int
    final_report: str


class GenerateSearchQueires(dspy.Signature):
    """Refine a claim into a list of search queries to verify its veracity."""

    claim: str = dspy.InputField(desc="The raw claim to be fact-checked.")  # pyright: ignore[reportInvalidTypeForm]
    context: str = dspy.InputField(  # pyright: ignore[reportInvalidTypeForm]
        desc="Previous search results to inform the query refinement."
    )
    queries: str = dspy.OutputField(
        desc="Exactly 3 distinct search queries to verify the claim, seperated by a semicolon. Do not include numbering."
    )  # pyright: ignore[reportInvalidTypeForm]


async def run_agent_debate(search_results, original_claim):
    model_client = OpenAIChatCompletionClient(
        model="gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    fact_checker = AssistantAgent(
        name="Factchecker",
        model_client=model_client,
        system_message="You are a sceptical fact-checker. Review the search results and highlight contradictions, weak sources, or logical leaps. Be critical.",
    )
    contextualist = AssistantAgent(
        name="Contextualist",
        model_client=model_client,
        system_message="You look for the nuance. Even if a claim is technically 'false', is there a part of it that is true? Provide background context.",
    )
    judge = AssistantAgent(
        name="Judge",
        model_client=model_client,
        system_message="Review the debate and provide a final Veracity Score (0-100) and summary.",
    )

    task = f"""
        The claim is: {original_claim}
        Here is the researched evidence: {search_results}\
        Fact-Checker and Contextualist: Please debate the validity of this claim
        based ONLY on the evidence provided.
        """

    debate = RoundRobinGroupChat(
        participants=[fact_checker, contextualist],
        termination_condition=MaxMessageTermination(max_messages=4),
    )
    debate_result = await debate.run(task=task)
    debate_transcript = "\n".join(m.content for m in debate_result.messages)

    judge_result = await judge.run(
        task=f"Here is the debate transcript:\n{debate_transcript}\nProvide your final Veracity Score and summary."
    )

    return judge_result.messages[-1].content


def research_planner(state: AgentState):
    print("PLANNING RESEARCH")
    planner = dspy.ChainOfThought(GenerateSearchQueires)
    result = planner(claim=state["claim"], context=state.get("search_results", []))
    raw_queries = result.queries
    query_list = [q.strip() for q in raw_queries.split(";") if q.strip()]
    if not query_list:
        query_list = [f"Fact check {state['claim']}"]
    return {"queries": query_list}


def search_engine(state: AgentState):
    print("SEARCHING")
    response = tavily_client.search(
        state["queries"][0],
        search_depth="advanced",
        max_results=3,
        include_raw_content=True,
    )
    new_results = [r["content"] for r in response["results"]]
    return {
        "search_results": state.get("search_results", []) + new_results,
        "revision_number": state.get("revision_number", 0) + 1,
    }


def sufficiency_checker(state: AgentState):
    print("CHECKING SUFFICIENCY")
    if state.get("revision_number", 0) > 3:
        return "sufficient"
    return "insufficient"


def final_report_node(state: AgentState):
    print("GENERATING FINAL REPORT")
    report = asyncio.run(run_agent_debate(state["search_results"], state["claim"]))
    return {"final_report": report}


workflow = StateGraph(AgentState)

workflow.add_node("planner", research_planner)
workflow.add_node("searcher", search_engine)
workflow.add_node("reporter", final_report_node)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "searcher")
workflow.add_edge("reporter", END)
workflow.add_conditional_edges(
    "searcher",
    sufficiency_checker,
    {
        "sufficient": "reporter",
        "insufficient": "searcher",
    },
)

app = workflow.compile()


if __name__ == "__main__":
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    initial_input = {"claim": "The moon is made of cheese."}
    result = app.invoke(initial_input)  # pyright: ignore[reportArgumentType]
    print(result)
