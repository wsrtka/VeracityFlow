from typing import AsyncGenerator
from uuid import uuid4

from autogen import MemorySaver
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.fact_checker import build_graph

app = FastAPI(title="VeracityFlow API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

checkpointer = MemorySaver()
graph = build_graph(checkpointer)


class ClaimRequest(BaseModel):
    claim: str


async def stream_pipeline(claim: str) -> AsyncGenerator[dict, None]:
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    async for event in graph.astream_events(
        {"claim": claim}, config=config, version="v2"
    ):
        kind = event["event"]
        name = event["name"]

        if kind == "on_chain_start" and name in ("planner", "searcher", "reporter"):
            node_labels = {
                "planner": "Planning research queries...",
                "searcher": "Searching for evidence...",
                "reporter": "Reporting results...",
            }
            yield {"event": "progress", "data": node_labels.get(name, name)}
        elif kind == "on_chain_end" and name == "reporter":
            output = event.get("data", {}).get("output", {})
            final_report = output.get("final_report", "")
            varacity_score = output.get("veracity_score", None)
            yield {
                "event": "result",
                "data": {
                    "report": final_report,
                    "score": varacity_score,
                    "thread_id": thread_id,
                },
            }

    yield {"event": "done", "data": {"thread_id": thread_id}}


@app.post("/check")
async def check_claim(request: ClaimRequest):
    async def generator():
        import json

        async for message in stream_pipeline(request.claim):
            yield {"event": message["event"], "data": json.dumps(message["data"])}

    return EventSourceResponse(generator(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}
