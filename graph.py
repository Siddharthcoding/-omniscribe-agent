"""LangGraph orchestration for the OmniScribe agent pipeline.

Flow:
  1. Structurer Agent -> generates cheat sheet from transcript
  2. Psychometrician Agent -> generates quiz from transcript + cheat sheet
  3. Save results to Supabase
"""

import os
import logging
import traceback
from dotenv import load_dotenv
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from agents.structurer import generate_cheat_sheet
from agents.psychometrician import generate_quiz
from supabase import create_client, Client

logger = logging.getLogger(__name__)
load_dotenv()

class AgentState(TypedDict):
    video_id: str
    full_text: str
    cheat_sheet: Optional[str]
    quiz_json: Optional[dict]
    error: Optional[str]


supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"],
)


async def structurer_node(state: AgentState) -> AgentState:
    vid = state["video_id"]
    logger.info("[structurer] [video=%s] Starting cheat sheet generation, text length=%d", vid, len(state["full_text"]))
    try:
        result = await generate_cheat_sheet(state["full_text"])
        cheat_sheet_md = _format_cheat_sheet(result)
        state["cheat_sheet"] = cheat_sheet_md
        logger.info("[structurer] [video=%s] Cheat sheet generated: title=%s, concepts=%d, glossary=%d",
                     vid, result.title, len(result.concepts), len(result.glossary))

        supabase.table("videos").update({
            "cheat_sheet": cheat_sheet_md,
        }).eq("id", vid).execute()
        logger.info("[structurer] [video=%s] Cheat sheet saved to Supabase", vid)

    except Exception as e:
        logger.error("[structurer] [video=%s] Failed: %s\n%s", vid, e, traceback.format_exc())
        state["error"] = f"Structurer failed: {str(e)}"

    return state


async def psychometrician_node(state: AgentState) -> AgentState:
    vid = state["video_id"]
    logger.info("[psychometrician] [video=%s] Starting quiz generation", vid)
    try:
        result = await generate_quiz(state["full_text"], state.get("cheat_sheet"))
        questions = [q.model_dump() for q in result.questions]
        state["quiz_json"] = {"questions": questions}
        logger.info("[psychometrician] [video=%s] Quiz generated: %d questions", vid, len(questions))

        supabase.table("quizzes").insert({
            "video_id": vid,
            "questions": questions,
        }).execute()
        logger.info("[psychometrician] [video=%s] Quiz saved to Supabase", vid)

    except Exception as e:
        logger.error("[psychometrician] [video=%s] Failed: %s\n%s", vid, e, traceback.format_exc())
        state["error"] = f"Psychometrician failed: {str(e)}"

    return state


async def finalize_node(state: AgentState) -> AgentState:
    vid = state["video_id"]
    if state.get("error"):
        logger.warning("[finalize] [video=%s] Pipeline had errors: %s. Setting status=failed", vid, state["error"])
        supabase.table("videos").update({
            "status": "failed",
            "error_message": state["error"],
        }).eq("id", vid).execute()
    else:
        logger.info("[finalize] [video=%s] Pipeline completed successfully. Setting status=completed", vid)
        supabase.table("videos").update({
            "status": "completed",
        }).eq("id", vid).execute()

    return state


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("structurer", structurer_node)
    builder.add_node("psychometrician", psychometrician_node)
    builder.add_node("finalize", finalize_node)

    builder.set_entry_point("structurer")
    builder.add_edge("structurer", "psychometrician")
    builder.add_edge("psychometrician", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


async def run_pipeline(video_id: str, full_text: str):
    logger.info("[pipeline] [video=%s] Starting agent pipeline", video_id)
    graph = build_graph()
    initial_state: AgentState = {
        "video_id": video_id,
        "full_text": full_text,
        "cheat_sheet": None,
        "quiz_json": None,
        "error": None,
    }
    await graph.ainvoke(initial_state)
    logger.info("[pipeline] [video=%s] Agent pipeline finished", video_id)


def _format_cheat_sheet(cs) -> str:
    lines = [f"# {cs.title}", ""]
    if cs.concepts:
        lines.append("## Key Concepts")
        for c in cs.concepts:
            lines.append(f"- {c}")
        lines.append("")
    if cs.glossary:
        lines.append("## Glossary")
        for g in cs.glossary:
            lines.append(f"- **{g.term}**: {g.definition}")
        lines.append("")
    if cs.formulas:
        lines.append("## Formulas & Patterns")
        for f in cs.formulas:
            lines.append(f"- {f}")
    return "\n".join(lines)
