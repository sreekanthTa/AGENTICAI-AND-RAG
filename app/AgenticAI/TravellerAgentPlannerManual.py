import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import httpx

# Load env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Groq client (OpenAI compatible)
client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"msg": "hello"}

# ----------------------------
# Traveller Agent Planner
# ----------------------------
class TravellerAgentPlanner:

    async def get_coords(self, city: str):
        url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json&limit=1"
        async with httpx.AsyncClient() as http:
            resp = await http.get(url)
            data = resp.json()
        if data:
            return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])}
        return None

    async def get_route(self, origin: str, destination: str):
        print("get_route is calling", origin, "=>", destination)

        origin_coords = await self.get_coords(origin)
        dest_coords = await self.get_coords(destination)

        if not origin_coords or not dest_coords:
            return f"Could not find coordinates for {origin} or {destination}."

        url = (
            f"https://router.project-osrm.org/route/v1/driving/"
            f"{origin_coords['lon']},{origin_coords['lat']};"
            f"{dest_coords['lon']},{dest_coords['lat']}?overview=false"
        )

        async with httpx.AsyncClient() as http:
            resp = await http.get(url)
            data = resp.json()

        if "routes" in data:
            route = data["routes"][0]
            return {
                "distance_km": round(route["distance"] / 1000, 2),
                "duration_hr": round(route["duration"] / 3600, 2),
            }
        return "Could not fetch route."

    async def get_weather(self, location: str):
        print("get weather is calling", location)
        url = f"https://wttr.in/{location}?format=j1"
        async with httpx.AsyncClient() as http:
            resp = await http.get(url)
            if resp.status_code == 200:
                return resp.json()["current_condition"][0]
        return f"Weather info not available for {location}"


# ----------------------------
# Define functions (tools)
# ----------------------------
functions = [
    {
        "name": "get_route",
        "description": "Get driving distance and duration between two cities.",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Starting city"},
                "destination": {"type": "string", "description": "Destination city"},
            },
            "required": ["origin", "destination"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get the weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"],
        },
    },
]

# ----------------------------
# Function that runs trip planning via Groq LLM
# ----------------------------
async def plan_trip_with_agent(user_message: str):
    planner = TravellerAgentPlanner()

    # Step 1: Ask LLM what to do
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are an expert travel planner. 
                Consider the Current Weather conditions and route options.
                Use these tools: get_weather for weather, get_route for routes.
                Always combine tool outputs into a natural plan.""",
            },
            {"role": "user", "content": user_message},
        ],
        functions=functions,
        function_call="auto",
    )

    msg = response.choices[0].message
    print("msg is", msg)

    # Step 2: If LLM decided to call a tool
    if msg.function_call:
        fn_name = msg.function_call.name
        args = json.loads(msg.function_call.arguments)

        # Call the correct local function
        if fn_name == "get_route":
            tool_result = await planner.get_route(args["origin"], args["destination"])
        elif fn_name == "get_weather":
            tool_result = await planner.get_weather(args["location"])
        else:
            tool_result = {"error": "Unknown function"}

        print("tool result ", tool_result)

        # Step 3: Send tool result back to LLM
        second_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert travel planner. 
                    Use the tool outputs to give a natural language plan.""",
                },
                {"role": "user", "content": user_message},
                msg,  # assistant function_call
                {
                    "role": "function",   # ✅ correct role
                    "name": fn_name,
                    "content": json.dumps(tool_result),
                },
            ],
        )
        return second_response.choices[0].message.content

    # Step 4: If no tool was called, just return the LLM’s reply
    return msg.content
