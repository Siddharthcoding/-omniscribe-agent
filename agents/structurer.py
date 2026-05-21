from langchain_core.prompts import ChatPromptTemplate
from gemini_client import get_llm
from schemas import CheatSheet

STRUCTURER_SYSTEM = """You are an expert technical writer and educator. Your job is to distill
video transcripts into clear, structured cheat sheets. Extract:
1. The main title/topic
2. Key concepts (in order of importance)
3. Important glossary terms with definitions
4. Any formulas, structural patterns, or frameworks mentioned

Output ONLY valid JSON matching the CheatSheet schema."""


async def generate_cheat_sheet(full_text: str) -> CheatSheet:
    llm = get_llm(temperature=0.3).with_structured_output(CheatSheet)

    prompt = ChatPromptTemplate.from_messages([
        ("system", STRUCTURER_SYSTEM),
        ("human", "Here is the transcript:\n\n{text}"),
    ])

    chain = prompt | llm
    result = await chain.ainvoke({"text": full_text})
    return result
