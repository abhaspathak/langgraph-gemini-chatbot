import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver


class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Create Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY
)


def chat_node(state: ChatbotState):

    # Take messages from state
    messages = state["messages"]

    # Send to Gemini
    response = llm.invoke(messages)

    # Store response in state
    return {
        "messages": [response]
    }


# Create Graph
checkpointer = MemorySaver()

graph = StateGraph(ChatbotState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)


# Compile graph
chatbot = graph.compile(
    checkpointer=checkpointer
)