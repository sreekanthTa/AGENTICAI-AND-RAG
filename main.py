import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client (OpenAI compatible)
client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"msg": "hello"}

# Use latest recommended Groq model
model_name = "llama-3.3-70b-versatile"

# ✅ Zero-shot Prompting
@app.post("/chat/zeroshot")
async def chat_zeroshot(req: ChatRequest):
    """
    Zero-shot prompting: No examples, only instruction + query.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an AI assistant that answers questions clearly."},
            {"role": "user", "content": req.message}
        ]
    )
    return {"response": response.choices[0].message.content}

# ✅ One-shot Prompting
@app.post("/chat/oneshot")
async def chat_oneshot(req: ChatRequest):
    """
    One-shot prompting: Provide one example before the query.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an AI expert translator."},
            {"role": "user", "content": "Translate 'hello' to Telugu."},
            {"role": "assistant", "content": "Namaste"},
            {"role": "user", "content": req.message}
        ]
    )
    return {"response": response.choices[0].message.content}

# ✅ Few-shot Prompting
@app.post("/chat/fewshots")
async def chat_fewshots(req: ChatRequest):
    """
    Few-shot prompting: Provide multiple examples.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an expert translator."},
            {"role": "user", "content": "Translate 'good morning' to French."},
            {"role": "assistant", "content": "Bonjour"},
            {"role": "user", "content": "Translate 'thank you' to French."},
            {"role": "assistant", "content": "Merci"},
            {"role": "user", "content": "Translate 'how are you?' to French."},
            {"role": "assistant", "content": "Comment ça va ?"},
            {"role": "user", "content": req.message}
        ]
    )
    return {"response": response.choices[0].message.content}

# ✅ Chain-of-Thought (CoT) Prompting
@app.get("/chat/cot")
async def chat_cot(query: str):
    """
    Chain-of-thought prompting: Show reasoning steps before the final answer.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system",
             "content": """You are a reasoning assistant.
             For each question, explain your reasoning step by step before giving the final answer.
             Example:
             Q: If there are 3 red balls and 2 blue balls, how many total?
             A: First count red balls (3), then blue balls (2). Add them: 3 + 2 = 5. Answer: 5."""},
            {"role": "user", "content": query}
        ]
    )
    return {"response": response.choices[0].message.content}

# ✅ Role-based Prompting
@app.get("/chat/role")
async def chat_role(query: str):
    """
    Role prompting: Assign the model a persona/role.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an expert teacher who simplifies any concept."},
            {"role": "user", "content": query}
        ]
    )
    return {"response": response.choices[0].message.content}

# ✅ Instruction Prompting: Here we get instructions from the user

# {
#   "message": "Summarize the following paragraph in 3 bullet points: FastAPI is a modern web framework for building APIs in Python..."
# }
@app.get("/chat/instruction")
async def chat_instruction(query:str):
    """
    Instruction prompting: Explicitly tell the model how to respond.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Follow instructions exactly as given."},
            {"role": "user", "content": f"Instruction: {query}"}
        ]
    )
    return {"response": response.choices[0].message.content}

# ✅ Self-Consistency Prompting
@app.get("/chat/self-consistency")
async def chat_self_consistency(query: str):
    """
    Self-consistency prompting: Ask model to solve with multiple reasoning paths.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system",
             "content": "Answer the question multiple times using different reasoning paths, then give the most common answer."},
            {"role": "user", "content": query}
        ]
    )
    return {"response": response.choices[0].message.content}

# ✅ ReAct (Reason + Act) Prompting
@app.get("/chat/react")
async def chat_react(query: str):
    """
    ReAct prompting: Mix reasoning and actions.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system",
             "content": "Think aloud about the problem (reason), then provide the final answer (act)."},
            {"role": "user", "content": query}
        ]
    )
    return {"response": response.choices[0].message.content}




@app.get("/plan_trip")
async def plan_trip(user_message: str):
    from app.AgenticAI.TravellerAgentPlanner import plan_trip_with_agent

    result = await plan_trip_with_agent(user_message)
    return {"result": result}