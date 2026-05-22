"""FastAPI entry point for the agent service.

Receives transcription results, runs the LangGraph pipeline,
and saves cheat sheets + quizzes to Supabase.
"""

import os
import re
import logging
import traceback
from datetime import datetime
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from graph import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [agent-service] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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
    logger.info("[pipeline] [video=%s] Pipeline request received, text length=%d", req.video_id, len(req.full_text))

    try:
        await run_pipeline(req.video_id, req.full_text)
        logger.info("[pipeline] [video=%s] Pipeline completed successfully", req.video_id)
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[pipeline] [video=%s] Pipeline failed: %s\n%s", req.video_id, e, traceback.format_exc())
        raise HTTPException(500, f"Pipeline failed: {str(e)}")


@app.post("/fetch-transcript", response_model=FetchTranscriptResponse)
async def fetch_transcript_endpoint(req: FetchTranscriptRequest, authorization: str | None = Header(None)):
    verify_auth(authorization)

    video_id = extract_video_id(req.url)
    if not video_id:
        logger.warning("[fetch-transcript] Invalid YouTube URL: %s", req.url)
        raise HTTPException(400, "Invalid YouTube URL")

    logger.info("[fetch-transcript] [video=%s] Fetching transcript for %s", video_id, req.url)
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        fetched = YouTubeTranscriptApi().fetch(video_id)
        segments = [
            {"text": s.text, "start": s.start, "end": s.start + s.duration}
            for s in fetched
        ]
        full_text = " ".join(s.text for s in fetched)
        logger.info("[fetch-transcript] [video=%s] Got %d segments, lang=%s", video_id, len(segments), fetched.language)
        return {"segments": segments, "full_text": full_text, "language": fetched.language}
    except Exception as e:
        logger.error("[fetch-transcript] [video=%s] Failed: %s", video_id, e)
        raise HTTPException(500, f"Transcript fetch failed: {e}")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
