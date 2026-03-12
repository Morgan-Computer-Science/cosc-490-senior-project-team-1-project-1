from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
import httpx
import os

app = FastAPI(title="ChatAI Backend", version="1.0.0")

origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
]

SYSTEM_PROMPT = "You are a helpful assistant."  # Customize this
MAX_HISTORY_TURNS = 20


# --- Models ---

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: list[Message] = []

class ChatResponse(BaseModel):
    reply: str
    updated_history: list[Message]


# --- Helpers ---

def to_gemini_history(history: list[Message]) -> list[dict]:
    """Convert internal history format to Gemini's 'contents' format."""
    role_map = {"user": "user", "assistant": "model"}
    return [
        {"role": role_map[m.role], "parts": [{"text": m.content}]}
        for m in history
    ]


# --- Routes ---

@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Build and trim history
    history = list(req.conversation_history)
    history = history[-(MAX_HISTORY_TURNS * 2):]

    # Append new user message
    history.append(Message(role="user", content=req.message))

    # Build Gemini contents payload
    contents = to_gemini_history(history)

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
    }

    # Try each model in order
    last_error = None
    async with httpx.AsyncClient(timeout=15.0) as client:
        for model in GEMINI_MODELS:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={GEMINI_API_KEY}"
            )
            try:
                response = await client.post(url, json=payload)
                data = response.json()

                if "candidates" in data:
                    reply = data["candidates"][0]["content"]["parts"][0]["text"]
                    history.append(Message(role="assistant", content=reply))
                    return ChatResponse(
                        reply=reply,
                        updated_history=history,
                    )

                last_error = data.get("error", {}).get("message", "Unknown error")

            except httpx.RequestError as e:
                last_error = str(e)

    raise HTTPException(status_code=502, detail=f"All Gemini models failed: {last_error}")