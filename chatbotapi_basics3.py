from fastapi import FastAPI
from pydantic import BaseModel
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    # Initialize model client
    model_client = OpenAIChatCompletionClient(
        model="gemini-1.5-flash-8b",
        api_key="AIzaSyCigOhKshdsZW8J1KH38W1n3pYwc5Isgvw"
    )

    # Prepare the user message with context
    user_msg = UserMessage(
        content=(
            f"You are an AI assistant for Harmanjeet Singh’s portfolio website. "
            "Reply for greetings like 'hello', 'hi', 'hey' with a friendly greeting. and ask how can you help. "
            "If the user asks about your capabilities, you can say: "
            "'I can answer questions about Harmanjeet Singh’s portfolio, including his experience, projects, and skills.'\n"
            "if the user says 'thank you' or 'thanks', respond with 'You're welcome! If you have any more questions, feel free to ask.'\n"
            "if user says 'who are you' or 'what is your name', respond with 'I am Harmanjeet Singh's portfolio assistant.'\n"
            "You MUST only answer questions using the resume below. "
            "If the question is not in the resume, reply: "
            "'That information is not available in the portfolio.'\n\n"
            f"Resume:\n{resume_text}\n\n"
            f"Question: {request.message}"
        ),
        source="user"
    )

    # Send message to the model
    response = await model_client.create([user_msg])

    # Close the client
    await model_client.close()

    # Extract assistant's reply
    reply_text = getattr(response, "content", str(response))

    return {"response": reply_text}
