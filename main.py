"""FastAPI entry point for the agent service.

Receives transcription results, runs the LangGraph pipeline,
and saves cheat sheets + quizzes to Supabase.
"""

import os
import re
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


class FetchTranscriptRequest(BaseModel):
    url: str


class FetchTranscriptResponse(BaseModel):
    segments: list
    full_text: str
    language: str


def verify_auth(authorization: str | None = Header(None)):
    if AUTH_TOKEN:
        if not authorization:
            raise HTTPException(401, "Missing Authorization header")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or token != AUTH_TOKEN:
            raise HTTPException(401, "Invalid authorization token")


def extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


@app.post("/process", response_model=ProcessResponse)
async def process_endpoint(req: ProcessRequest, authorization: str | None = Header(None)):
    verify_auth(authorization)

    try:
        await run_pipeline(req.video_id, req.full_text)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(500, f"Pipeline failed: {str(e)}")


@app.post("/fetch-transcript", response_model=FetchTranscriptResponse)
async def fetch_transcript_endpoint(req: FetchTranscriptRequest, authorization: str | None = Header(None)):
    verify_auth(authorization)

    video_id = extract_video_id(req.url)
    if not video_id:
        raise HTTPException(400, "Invalid YouTube URL")

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        if transcript:
            segments = [
                {"text": s["text"], "start": s["start"], "end": s["start"] + s["duration"]}
                for s in transcript
            ]
            full_text = " ".join(s["text"] for s in transcript)
            return {"segments": segments, "full_text": full_text, "language": "en"}
    except Exception:
        pass

    raise HTTPException(500, "Could not fetch transcript")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
