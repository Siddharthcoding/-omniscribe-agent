from langchain_core.prompts import ChatPromptTemplate
from gemini_client import get_llm
from schemas import Quiz

PSYCHOMETRICIAN_SYSTEM = """You are an expert educational psychometrician. Your job is to create
balanced multiple-choice questions from video content.

Rules:
- Create exactly 5 questions
- Mix difficulty levels: 2 easy, 2 medium, 1 hard
- Each question must have exactly 4 options
- The correct answer index (0-3) must be accurate
- Questions should test comprehension, not trivia
- Use the cheat sheet concepts to focus on important material

Output ONLY valid JSON matching the Quiz schema."""


async def generate_quiz(full_text: str, cheat_sheet: str | None = None) -> Quiz:
    llm = get_llm(temperature=0.4).with_structured_output(Quiz)

    context = full_text
    if cheat_sheet:
        context = f"Cheat Sheet:\n{cheat_sheet}\n\nFull Transcript:\n{full_text}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", PSYCHOMETRICIAN_SYSTEM),
        ("human", "Create a quiz based on this content:\n\n{text}"),
    ])

    chain = prompt | llm
    result = await chain.ainvoke({"text": context})
    return result
