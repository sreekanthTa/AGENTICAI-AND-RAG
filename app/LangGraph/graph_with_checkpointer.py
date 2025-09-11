from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

# Import LangChain message classes
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# JSON-based checkpointer for persistent memory
import json

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Groq client (OpenAI compatible)
llm = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


# -----------------------------
# Persistent Checkpointer
# -----------------------------
class FileCheckpointer:
    def __init__(self, path="chat_memory.json"):
        self.path = path
        self.state = {}
        if os.path.exists(path):
            with open(path, "r") as f:
                self.state = json.load(f)

    def save(self, key, value):
        self.state[key] = value
        with open(self.path, "w") as f:
            json.dump(self.state, f)

    def load(self, key):
        return self.state.get(key, None)


checkpointer = FileCheckpointer()


# -----------------------------
# State Definition
# -----------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    classification: str


# -----------------------------
# Helper to convert messages
# -----------------------------
def convert_messages(state: State):
    converted = []
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            role, content = "user", msg.content
        elif isinstance(msg, AIMessage):
            role, content = "assistant", msg.content
        elif isinstance(msg, SystemMessage):
            role, content = "system", msg.content
        elif isinstance(msg, dict):
            role, content = msg.get("role", "user"), msg.get("content")
        else:
            role, content = "user", str(msg)
        converted.append({"role": role, "content": content})
    return converted


# -----------------------------
# Node Functions
# -----------------------------
def classify_question(state: State):
    print("\n\n[Classify Question]")
    messages = convert_messages(state)

    classification_prompt = {
        "role": "system",
        "content": (
            "Classify the user's query strictly as either 'math' or 'science'. "
            "If it's neither, reply with 'normal'. Reply with only one word."
        ),
    }

    response = llm.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[classification_prompt] + messages,
        temperature=0,
    )

    classification = response.choices[0].message.content.strip().lower()
    print(f"[Classification Result] → {classification}")

    # Save to persistent memory
    checkpointer.save("last_classification", classification)
    return {"classification": classification}


def route_by_classification(state: State):
    classification = state.get("classification", "")
    if classification == "math":
        return "answer_math"
    elif classification == "science":
        return "answer_science"
    else:
        return "answer_normal"


def generate_math_answer(state: State):
    print("\n\n[Generate Math Answer]")
    messages = convert_messages(state)
    response = llm.chat.completions.create(
        model="llama-3.3-70b-versatile", messages=messages, temperature=0.7
    )
    return {"messages": [{"role": "assistant", "content": response.choices[0].message.content}]}


def generate_science_answer(state: State):
    print("\n\n[Generate Science Answer]")
    messages = convert_messages(state)
    response = llm.chat.completions.create(
        model="llama-3.3-70b-versatile", messages=messages, temperature=0.7
    )
    return {"messages": [{"role": "assistant", "content": response.choices[0].message.content}]}


def generate_normal_answer(state: State):
    print("\n\n[Generate Normal Answer]")
    messages = convert_messages(state)
    response = llm.chat.completions.create(
        model="llama-3.3-70b-versatile", messages=messages, temperature=0.7
    )
    return {"messages": [{"role": "assistant", "content": response.choices[0].message.content}]}


# -----------------------------
# Build Graph
# -----------------------------
graph_builder = StateGraph(State)
graph_builder.add_node("classify", classify_question)
graph_builder.add_node("answer_math", generate_math_answer)
graph_builder.add_node("answer_science", generate_science_answer)
graph_builder.add_node("answer_normal", generate_normal_answer)

graph_builder.add_edge(START, "classify")
graph_builder.add_conditional_edges("classify", route_by_classification)
graph_builder.add_edge("answer_math", END)
graph_builder.add_edge("answer_science", END)
graph_builder.add_edge("answer_normal", END)

graph = graph_builder.compile()  # Checkpointer already saves state separately


# -----------------------------
# REPL / Stream Updates
# -----------------------------
def stream_graph_updates(user_input: str):
    # Load previous messages from memory
    previous_messages = checkpointer.load("conversation") or []

    input_state = {"messages": previous_messages + [{"role": "user", "content": user_input}]}
    for event in graph.stream(input_state):
        for value in event.values():
            if "messages" in value:
                last_message = value["messages"][-1]
                print(f"Assistant: {last_message['content']}")

                # Persist conversation
                checkpointer.save("conversation", input_state["messages"] + [{"role": "assistant", "content": last_message["content"]}])


# -----------------------------
# Run Chat
# -----------------------------
if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
