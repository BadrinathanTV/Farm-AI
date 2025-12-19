# app.py

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from graph import app
from core.profile_manager import ProfileManager
from core.chat_history_manager import ChatHistoryManager
import uuid

# --- Page & State Configuration ---
st.set_page_config(page_title="Farm AI Assistant", page_icon="👨‍🌾", layout="wide")

def initialize_session_state():
    """Initializes all necessary session state variables."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = []
    
    # Session state for managing file uploader reset
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    
    # Initialize managers once
    if "profile_manager" not in st.session_state:
        st.session_state.profile_manager = ProfileManager()
    if "chat_history_manager" not in st.session_state:
        st.session_state.chat_history_manager = ChatHistoryManager()

# --- Authentication Logic ---
def show_login_signup_page():
    """Displays the login and sign-up forms."""
    st.title("Welcome to the Farm AI Assistant 👨‍🌾")
    
    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login")
            if submitted:
                user = st.session_state.profile_manager.authenticate_user(username, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user.user_id
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

    with signup_tab:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a Username", key="signup_username")
            new_password = st.text_input("Choose a Password", type="password", key="signup_password")
            submitted = st.form_submit_button("Sign Up")
            if submitted:
                try:
                    st.session_state.profile_manager.create_user(new_username, new_password)
                    st.success("Account created! Please login.")
                except ValueError as e:
                    st.error(e)

# --- Main Chat Interface Logic ---
def show_chat_interface():
    """Displays the main chat interface with sidebar for session management."""
    # --- Sidebar for Chat Session Management ---
    with st.sidebar:
        st.header(f"Welcome, {st.session_state.user_id}!")
        
        if st.button("➕ New Chat"):
            st.session_state.chat_id = str(uuid.uuid1())
            st.session_state.messages = []
            st.rerun()

        st.subheader("Chat History")
        st.session_state.chat_sessions = st.session_state.chat_history_manager.get_chat_sessions(st.session_state.user_id)
        
        for session in st.session_state.chat_sessions:
            if st.button(session["title"], key=session["chat_id"]):
                st.session_state.chat_id = session["chat_id"]
                st.session_state.messages = st.session_state.chat_history_manager.load_history(session["chat_id"])
                st.rerun()
        
        if st.button("Logout"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

    # --- Main Chat Window ---
    st.title("💬 Chat with your AI Assistant")
    
    if st.session_state.chat_id is None:
        st.info("Select a chat from the history or start a new one!")
        return

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message.type, avatar="🧑‍💻" if message.type == "human" else "👨‍🌾"):
            st.markdown(message.content)

    # --- Input Area ---
    # We use a popover for image upload to mimic a "paperclip" attachment style
    with st.popover("📎 Add Image", help="Upload a plant image for diagnosis"):
        # Use a dynamic key to allow clearing the uploader
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"], 
            key=f"uploader_{st.session_state.uploader_key}"
        )

    # Chat Input - Always rendered
    if prompt := st.chat_input("Ask a question or describe your plant issue..."):
        user_input = prompt
        image_bytes = None
        
        # Check for uploaded image
        if uploaded_file:
            image_bytes = uploaded_file.getvalue()
            # Append a note about the image to the user input if not already explicit
            if "image" not in user_input.lower():
                user_input += " [Image Uploaded]"

        # Display User Message immediately
        with st.chat_message("human", avatar="🧑‍💻"):
            if image_bytes:
                st.image(uploaded_file, caption="Uploaded Image", width=300)
            st.markdown(prompt)

        # Add to history
        st.session_state.messages.append(HumanMessage(content=user_input))

        # Invoke graph and display response
        with st.chat_message("ai", avatar="👨‍🌾"):
            with st.spinner("Assistant is thinking..."):
                payload = {
                    "user_id": st.session_state.user_id, 
                    "messages": st.session_state.messages,
                    "image_data": image_bytes 
                }
                
                try:
                    response = app.invoke(payload)
                    ai_response = response["messages"][-1]
                    st.markdown(ai_response.content)
                    st.session_state.messages.append(ai_response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        # Save the updated history
        st.session_state.chat_history_manager.save_history(st.session_state.user_id, st.session_state.chat_id, st.session_state.messages)
        
        # RESET UPLOADER: Increment key to clear the file uploader for the next turn
        if uploaded_file:
            st.session_state.uploader_key += 1
            st.rerun()


# --- Application Entry Point ---
initialize_session_state()

if st.session_state.logged_in:
    show_chat_interface()
else:
    show_login_signup_page()

