# Standard library imports
from langgraph.checkpoint.sqlite import SqliteSaver
from logging import config
import os
import time

# Third-party library imports
from narwhals import col
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
import torch
import pandas as pd

from langchain_core.language_models.chat_models import BaseChatModel


from langchain_ollama import ChatOllama

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph, START, END

import sqlite3


# Load environment variables from .env file
load_dotenv()


# set up llm workflow and load it if it is available
# Load messages from external database


# ------ Environment variables ------
PAGE_TITLE = "Chat Assistant"

DB_PATH = "state_db/chat_history.db"

# ------ Set up database ------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
import threading

# Create a thread-local storage for SQLite connections
thread_local = threading.local()

def get_thread_local_connection():
    if not hasattr(thread_local, "conn"):
        thread_local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return thread_local.conn

memory = SqliteSaver(conn=get_thread_local_connection())
# ------ Chatbot ------
chat_model = ChatOllama(model="gemma3:1b",
                        temperature=0.0,
                        verbose=True)


class State(MessagesState):
    """
    Represents the state of the conversation, including a summary of the dialogue.
    Inherits from MessagesState to manage message history.
    """
    summary: str  # Additional state key


def call_model(state: State):
    summary = state.get("summary", "")

    #  If there is any summary, we add it to the message
    if summary:

        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"
    try:
        response = chat_model.invoke(messages)
        return {"messages": response}
    except Exception as e:
        error_message = f"An error occurred while invoking the chat model: {str(e)}"
        print(error_message)
        return {"messages": [AIMessage(content=error_message)]}
        messages = [SystemMessage(content=system_message)] + state["messages"]

    else:
        messages = state["messages"]

    print(type(messages))
    print(messages)
    response = chat_model.invoke(messages)
    return {"messages": response}


def summarize_conversation(state: State):

    summary = state.get('summary', "")

    if summary:

        summary_message = (
            f"This is a summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )

    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = chat_model.invoke(messages)

    # Keep only the last two messages and delete all previous
    last_two_messages = [RemoveMessage(id=m.id)
                         for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": last_two_messages}


def should_continue(state: State):
    """Return the next note to execute."""
    # Either summarize, if there are so many messages, or skip it and end it.

    messages = state["messages"]

    if len(messages) > 6:
        return "summarize_conversation"
    return END


#  Create workflow and checkpoint
workflow = StateGraph(State)

workflow.add_node("conversation", call_model)
workflow.add_node("summarize_conversation", summarize_conversation)

workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# Create graph and memory to save conversation
# graph = workflow.compile(checkpointer=memory) # Memory by using the sqlite database
if "graph" not in st.session_state:
    st.session_state["graph"] = workflow.compile(checkpointer=memory)

chat_config = {"configurable": {"thread_id": "1"}}


# ------ Set up UI ------
st.set_page_config(page_title=PAGE_TITLE)
col1, col2, col3 = st.columns([1, 25, 1])

with col2:
    st.title(PAGE_TITLE)

# --- Initialize Session State for Messages ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- Display Previous Messages ---
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.write(message.content)

# --- Handle User Input ---
user_query = st.chat_input("Ask me anything ..")

if user_query:
    # Display user message
    with st.chat_message("user"):
        st.write(user_query)

    # Display assistant response
    with st.chat_message("assistant"):
        with st.status("Generating 💡...", expanded=True):
            result = st.session_state["graph"].invoke(
                {"messages": HumanMessage(user_query)},
                chat_config,
            )
            message_placeholder = st.empty()
            message_placeholder.markdown(result["messages"][-1].content)

    st.session_state.messages.extend(
        [HumanMessage(content=user_query), AIMessage(
            content=result["messages"][-1].content)])
