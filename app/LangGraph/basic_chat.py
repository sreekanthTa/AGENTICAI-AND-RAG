from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Image, display

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Groq client (OpenAI compatible)
llm = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

# Define the state type
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Chatbot function
def chatbot(state: State):
    messages = []
    print("mnessages", state["messages"])
    for msg in state["messages"]:
        print("msg is",msg)
        # If msg is already a dict with role + content
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            messages.append({"role": msg["role"], "content": msg["content"]})
        # If msg is a LangGraph BaseMessage-like object
        elif hasattr(msg, "role") and hasattr(msg, "content"):
            messages.append({"role": msg.role, "content": msg.content})
        # If msg only has content, assume it's from user
        elif hasattr(msg, "content"):
            messages.append({"role": "user", "content": msg.content})
        else:
            raise ValueError(f"Cannot convert message to dict: {msg}")

    # Call Groq/OpenAI API
    response = llm.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Replace with your desired model
        messages=messages,
        temperature=0.7
    )

    # Return assistant message in expected format
    return {
        "messages": [
            {"role": "assistant", "content": response.choices[0].message.content}
        ]
    }

# Build LangGraph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# Optional: visualize the graph
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"Graph visualization skipped: {e}")

# Streaming helper function
def stream_graph_updates(user_input: str):
    # result = graph.stream({"messages": [{"role": "user", "content": user_input}]})
    # print("result is", result.values())
    # return
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        print("even tis", event)
        for value in event.values():
            print("Assistant:", value["messages"][-1]["content"])

# Interactive loop
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except Exception as e:
        print(f"Error occurred: {e}")
        # Fallback input if input() fails
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
