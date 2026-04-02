from fastapi import FastAPI
from pydantic import BaseModel
from openai import AsyncOpenAI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://harmanjeet3.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load resume once
with open("Harman_Resume_QA_.txt", "r", encoding="utf-8", errors="ignore") as f:
    resume_text = f.read()

class UserRequest(BaseModel):
    message: str

@app.post("/ask")
async def ask_resume(request: UserRequest):
    client = AsyncOpenAI(
        api_key="AIzaSyAXexY48Ezd0I-g8siYw7aplfKz-iwH55k",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    response = await client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[
            {
                "role": "user",
                "content": (
                    "You are an AI assistant for Harmanjeet Singh's portfolio website. "
                    "Reply for greetings like 'hello', 'hi', 'hey' with a friendly greeting. and ask how can you help. "
                    "If the user asks about your capabilities, you can say: "
                    "'I can answer questions about Harmanjeet Singh's portfolio, including his experience, projects, and skills.'\n"
                    "if the user says 'thank you' or 'thanks', respond with 'You are welcome! If you have any more questions, feel free to ask.'\n"
                    "if user says 'who are you' or 'what is your name', respond with 'I am Harmanjeet Singh portfolio assistant.'\n"
                    "You MUST only answer questions using the resume below. "
                    "If the question is not in the resume, reply: "
                    "'That information is not available in the portfolio.'\n\n"
                    f"Resume:\n{resume_text}\n\n"
                    f"Question: {request.message}"
                )
            }
        ]
    )

    reply_text = response.choices[0].message.content
    return {"response": reply_text}