from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import anthropic
import os

app = FastAPI(title="AI Backend", version="1.0.0")

# CORS — update origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SYSTEM_PROMPT = "You are a helpful assistant."  # Customize this


# --- Models ---

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[list[dict]] = []

class ChatResponse(BaseModel):
    reply: str
    updated_history: list[dict]

# --- Routes ---

@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Append new user message to history
    history = req.conversation_history + [
        {"role": "user", "content": req.message}
    ]

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=history,
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {str(e)}")

    assistant_reply = response.content[0].text

    # Append assistant reply so caller can maintain history
    updated_history = history + [
        {"role": "assistant", "content": assistant_reply}
    ]

    return ChatResponse(reply=assistant_reply, updated_history=updated_history)