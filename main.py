"""FastAPI entry point for the agent service.

Receives transcription results, runs the LangGraph pipeline,
and saves cheat sheets + quizzes to Supabase.
"""

import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from graph import run_pipeline

app = FastAPI(title="OmniScribe Agent Service")

AUTH_TOKEN = os.environ.get("OMNISCRIBE_AUTH_TOKEN", "")


class ProcessRequest(BaseModel):
    video_id: str
    full_text: str


class ProcessResponse(BaseModel):
    status: str


def verify_auth(authorization: str | None = Header(None)):
    if AUTH_TOKEN:
        if not authorization:
            raise HTTPException(401, "Missing Authorization header")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or token != AUTH_TOKEN:
            raise HTTPException(401, "Invalid authorization token")


@app.post("/process", response_model=ProcessResponse)
async def process_endpoint(req: ProcessRequest, auth=Header(None)):
    verify_auth(auth)

    try:
        await run_pipeline(req.video_id, req.full_text)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(500, f"Pipeline failed: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok"}
