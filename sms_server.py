
from fastapi import FastAPI, Form, Response
from twilio.twiml.messaging_response import MessagingResponse
import uvicorn
import os
from typing import Annotated
from langchain_core.messages import HumanMessage
import uuid

# Import existing Farm-AI components
# Ensure these imports work based on your project structure
from graph import app as agent_app
from core.profile_manager import ProfileManager
from core.chat_history_manager import ChatHistoryManager

# Initialize FastAPI
app = FastAPI(title="Farm-AI SMS Server")

# Initialize Managers
profile_manager = ProfileManager()
chat_history_manager = ChatHistoryManager()

@app.post("/sms")
async def reply_to_sms(Body: Annotated[str, Form()], From: Annotated[str, Form()], To: Annotated[str, Form()]):
    """
    Handle incoming SMS from Twilio.
    """
    user_message = Body
    sender_phone = From
    # Use the phone number as the User ID
    user_id = sender_phone 
    
    print(f"Received SMS from {user_id}: {user_message}")

    # 1. User Management
    # Check if user exists (by phone number)
    user = profile_manager.get_user(user_id)
    if not user:
        try:
            # Create a new user for this phone number with a default password
            # In a real app, we might handle this differently (e.g. OTP)
            print(f"Creating new user for {user_id}")
            # Define a default password for SMS users or random
            default_password = "sms_user_password" 
            profile_manager.create_user(user_id, default_password)
            user = profile_manager.get_user(user_id)
        except Exception as e:
            print(f"Error creating/retrieving user {user_id}: {e}")
            # Proceeding might fail if user is None, but let's try invoking the graph anyway
            # The graph expects a user_id string mostly.

    # 2. Chat Session Management
    # Retrieve existing sessions or create a new one
    sessions = chat_history_manager.get_chat_sessions(user_id)
    if sessions:
        # Use the most recent session
        # Ideally, we might want to check if the last message was recent to decide 
        # seamlessly continuing vs starting new, but simple is better here.
        chat_id = sessions[0]["chat_id"]
        messages = chat_history_manager.load_history(chat_id)
    else:
        # Create a new session
        chat_id = str(uuid.uuid1())
        messages = []
        # Note: ChatHistoryManager doesn't have an explicit 'create_session' 
        # it just saves history.
    
    # 3. Invoke Agent Graph
    # Append user message
    messages.append(HumanMessage(content=user_message))
    
    payload = {
        "user_id": user_id,
        "messages": messages,
        "image_data": None # SMS currently only text
    }
    
    ai_text = "Sorry, I'm having trouble connecting to the farm brain right now."
    
    try:
        # Invoke the LangGraph app
        response = agent_app.invoke(payload)
        
        # Get the AI's response
        if response and "messages" in response and response["messages"]:
            ai_response_message = response["messages"][-1]
            ai_text = ai_response_message.content
            
            # Save the updated history (User + AI)
            messages.append(ai_response_message)
            chat_history_manager.save_history(user_id, chat_id, messages)
        else:
            ai_text = "I didn't get a response from the system."

    except Exception as e:
        print(f"Error invoking agent: {e}")
        ai_text = f"Error processing your request: {str(e)}"

    # 4. Send Response via Twilio
    resp = MessagingResponse()
    resp.message(ai_text)
    
    return Response(content=str(resp), media_type="application/xml")

if __name__ == "__main__":
    # Run on port 8001 to avoid conflict with MCP server (8000)
    uvicorn.run(app, host="0.0.0.0", port=8001)
