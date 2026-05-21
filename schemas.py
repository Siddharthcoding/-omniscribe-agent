from pydantic import BaseModel, Field


class GlossaryTerm(BaseModel):
    term: str = Field(description="The glossary term or keyword")
    definition: str = Field(description="Short definition of the term")


class CheatSheet(BaseModel):
    title: str = Field(description="Title of the cheat sheet")
    concepts: list[str] = Field(description="Key concepts distilled from the video")
    glossary: list[GlossaryTerm] = Field(description="Important glossary terms")
    formulas: list[str] = Field(description="Any formulas or structural patterns mentioned")


class MCQQuestion(BaseModel):
    question: str = Field(description="The multiple-choice question")
    options: list[str] = Field(description="4 answer options", min_length=4, max_length=4)
    correct_index: int = Field(description="Index (0-3) of the correct answer in options")
    difficulty: str = Field(description="Difficulty: easy, medium, or hard")


class Quiz(BaseModel):
    questions: list[MCQQuestion] = Field(description="5 balanced MCQ questions", min_length=5, max_length=5)
