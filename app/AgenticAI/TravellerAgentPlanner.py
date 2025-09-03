import os
import json
import logging
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ----------------------------
# Setup Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------
# Setup FastAPI and Environment
# ----------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is missing.")

app = FastAPI(title="Trip Planner API", description="API for planning trips with weather and route info.")

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"msg": "Welcome to the Trip Planner API. Use /plan_trip to interact with the agent."}

# ----------------------------
# Tools (route + weather)
# ----------------------------
@tool
def get_coords(city: str) -> dict | None:
    """Get lat/lon from OpenStreetMap for a city."""
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json&limit=1"
        headers = {"User-Agent": "TripPlannerAPI/1.0"}
        resp = requests.get(url, headers=headers, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data:
            return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])}
        logger.warning(f"No coordinates found for city: {city}")
        return None
    except requests.RequestException as e:
        logger.error(f"Error fetching coordinates for {city}: {e}")
        return None

@tool
def get_route(origin: str, destination: str) -> dict | str:
    """Driving route between two cities."""
    try:
        # Basic check for international trips
        if ("UK" in destination or "United Kingdom" in destination) and ("India" in origin or "Kadapa" in origin):
            logger.info(f"Detected international trip from {origin} to {destination}")
            return "Driving route not feasible for international travel. Consider flights."
        
        origin_coords = get_coords.invoke({"city": origin})
        dest_coords = get_coords.invoke({"city": destination})
        if not origin_coords or not dest_coords:
            logger.warning(f"Could not find coordinates for {origin} or {destination}")
            return f"Could not find coordinates for {origin} or {destination}."
        
        url = (
            f"https://router.project-osrm.org/route/v1/driving/"
            f"{origin_coords['lon']},{origin_coords['lat']};"
            f"{dest_coords['lon']},{dest_coords['lat']}?overview=false"
        )
        headers = {"User-Agent": "TripPlannerAPI/1.0"}
        resp = requests.get(url, headers=headers, timeout=5).json()
        if "routes" in resp:
            route = resp["routes"][0]
            return {
                "distance_km": round(route["distance"] / 1000, 2),
                "duration_hr": round(route["duration"] / 3600, 2),
            }
        logger.warning(f"No routes found for {origin} to {destination}")
        return "Could not fetch route."
    except requests.RequestException as e:
        logger.error(f"Error fetching route from {origin} to {destination}: {e}")
        return "Could not fetch route."

@tool
def get_weather(location: str) -> dict | str:
    """Get current weather for a city."""
    try:
        url = f"https://wttr.in/{location}?format=j1"
        headers = {"User-Agent": "TripPlannerAPI/1.0"}
        resp = requests.get(url, headers=headers, timeout=5)
        resp.raise_for_status()
        return resp.json()["current_condition"][0]
    except requests.RequestException as e:
        logger.error(f"Error fetching weather for {location}: {e}")
        return f"Weather info not available for {location}"

# ----------------------------
# Agent (LLM + Tools)
# ----------------------------
try:
    llm = ChatOpenAI(
        model="llama-3.3-70b-versatile",  # Verify model availability with Groq
        temperature=0,
        openai_api_key=api_key,
        openai_api_base=os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1"),
    )
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    raise ValueError(f"Failed to initialize LLM: {e}")

tools = [get_coords, get_route, get_weather]
agent = create_react_agent(llm, tools)

# //Service Method
# //Service Method
# //Service Method
# Service method
# Service method
async def plan_trip_with_agent(user_message: str):
    try:
        logger.info(f"Received request with user_message: {user_message}")

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_message}]}
        )

        # Extract last AI message content
        response_text = ""
        if hasattr(result, "messages"):
            # Filter AI messages only
            ai_messages = [m for m in result.messages if getattr(m, "type", "") == "ai"]
            if ai_messages:
                response_text = ai_messages[-1].content.strip()
        elif hasattr(result, "content"):
            response_text = result.content.strip()
        else:
            response_text = str(result)

        # Return human-readable plan
        return {"plan": response_text}

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
