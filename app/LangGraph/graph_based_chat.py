from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

# Import LangChain message classes
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Groq client (OpenAI compatible)
llm = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


class State(TypedDict):
    messages: Annotated[list, add_messages]
    classification: str


def convert_messages(state: State):
    """Convert LangChain message objects to OpenAI-compatible format."""
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


def classify_question(state: State):
    print("\n\n[Classify Question]")
    messages = convert_messages(state)

    classification_prompt = {
        "role": "system",
        "content": (
            "Classify the user's query strictly as one of the following categories:\n"
            "- math\n- science\n- normal\n\n"
            "Return only a valid JSON object: {\"category\": \"math\"}"
        )
    }

    response = llm.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[classification_prompt] + messages,
        temperature=0
    )

    raw_output = response.choices[0].message.content.strip()
    print(f"[Raw Classification] → {raw_output}")

    try:
        parsed = json.loads(raw_output)
        first_classification = parsed.get("category", "normal").lower()
    except json.JSONDecodeError:
        print("[Warning] Classification not in JSON format, defaulting to normal")
        first_classification = "normal"

    # Refinement step: pass to a cheaper/faster model for final decision
    refinement_prompt = {
        "role": "system",
        "content": (
            f"You are a strict router. The first classifier said '{first_classification}'. "
            "Respond ONLY with one of: 'math', 'science', 'normal'. "
            "Decide based on the user's question and first classifier's output."
        )
    }

    refinement_response = llm.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Faster model for routing
        messages=[refinement_prompt] + messages,
        temperature=0
    )

    refined_output = refinement_response.choices[0].message.content.strip().lower()
    print(f"[Refined Classification] → {refined_output}")

    # //It automatically adds to the state 
    return {"classification": refined_output}


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
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7
    )
    return {"messages": [{"role": "assistant", "content": response.choices[0].message.content}]}


def generate_science_answer(state: State):
    print("\n\n[Generate Science Answer]")
    messages = convert_messages(state)
    response = llm.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7
    )
    return {"messages": [{"role": "assistant", "content": response.choices[0].message.content}]}


def generate_normal_answer(state: State):
    print("\n\n[Generate Normal Answer]")
    messages = convert_messages(state)
    response = llm.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7
    )
    return {"messages": [{"role": "assistant", "content": response.choices[0].message.content}]}


# Build graph
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

graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    input_state = {"messages": [{"role": "user", "content": user_input}]}
    for event in graph.stream(input_state):
        for value in event.values():
            if "messages" in value:
                last_message = value["messages"][-1]
                print(f"Assistant: {last_message['content']}")


if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)

