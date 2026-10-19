"""Example: plug the bot into a FastAPI chatbot.   uvicorn examples.fastapi_hook:app --port 8001
In your own app you only need the marked lines. The similarity report prints in the terminal."""
from fastapi import FastAPI
from pydantic import BaseModel

from qsim.bot import QABot                          # (1) import

app = FastAPI()
bot = QABot()                                        # (2) once at startup: loads the model + data/knowledge_base.json


class ChatRequest(BaseModel):
    question: str


@app.post("/api/chat")
def chat(req: ChatRequest):
    turn = bot.ask(req.question)                     # (3) saves the question, checks similarity, picks the answer
    return {"answer": turn.answer, "source": turn.source}
