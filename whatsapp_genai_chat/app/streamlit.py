import os

import requests
import streamlit as st
from streamlit_chat import message

# API URL
API_URL = "http://127.0.0.1:8000"

# Define user
response = requests.get(f"{API_URL}/get_user")
user = response.json()["user"]
user_file = response.json()["user_file"]

# Streamlit app setup
st.set_page_config(
    page_title=user,
    page_icon="🌈",
    layout="centered",
)

# Styling
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to bottom, #E3F2FD, #FAFAFA);
        font-family: 'Roboto', sans-serif;
    }
    .stButton > button {
        background-color: #03A9F4;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stTextInput > div > input {
        border: 2px solid #BBDEFB;
        border-radius: 10px;
        padding: 10px;
    }
    .message { 
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        font-size: 16px;
    }
    .message.user {
        background-color: #BBDEFB;
        text-align: right;
    }
    .message.bot {
        background-color: #F5F5F5;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Log file path
LOG_FILE = f"data/{user_file}/wc_logs.txt"


# Function to log chat history
def log_chat_history():
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        for msg in st.session_state.messages[-2:]:  # Log the last user-bot exchange
            role = "User" if msg["role"] == "user" else "Bot"
            log_file.write(f"{role}: {msg['content']}\n")
        log_file.write("---\n")


# Session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context" not in st.session_state:
    st.session_state.context = []

# Title
st.title(user)
st.markdown(f"Chat with {user}")

# Chat input
user_input = st.text_input("Enter your message:", "", placeholder="Type here...")
if st.button("Send") and user_input:
    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Make API call
    try:
        response = requests.post(
            f"{API_URL}/get_response", json={"question": user_input}
        )
        response.raise_for_status()
        bot_response = response.json().get("response", "Oops, no response!")
        retrieved_context = response.json().get("retrieved_docs", [])
        st.session_state.messages.append({"role": "bot", "content": bot_response})
        st.session_state.context = retrieved_context

        # Log chat history
        log_chat_history()
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Display context in a collapsible box
with st.expander("Context for the conversation"):
    if st.session_state.context:
        for idx, doc in enumerate(st.session_state.context):
            st.markdown(f"**Context {idx + 1}:** {doc}")
    else:
        st.markdown("No context available yet.")

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=f"user-{msg}")
    else:
        message(msg["content"], is_user=False, key=f"bot-{msg}")
