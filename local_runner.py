# local_runner.py

import os
import sys
import asyncio
from dotenv import load_dotenv
import vertexai
import google.auth
from google.auth.transport.requests import Request
from google.adk.agents import LlmAgent
from google.adk.tools.application_integration_tool.application_integration_toolset import ApplicationIntegrationToolset
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.genai import types as genai_types

# 1️⃣ Load environment variables
load_dotenv()
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
CONNECTION_NAME = os.getenv("CONNECTION_NAME")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

if not all([PROJECT_ID, LOCATION, CONNECTION_NAME, GEMINI_MODEL]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

# 2️⃣ Configure Vertex AI Authentication
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
except Exception as e:
    print(f"❌ Failed to initialize Vertex AI: {e}")
    sys.exit(1)

# 3️⃣ Create ServiceNow toolset
def create_servicenow_toolset():
    servicenow_toolset = ApplicationIntegrationToolset(
        project=PROJECT_ID,
        location=LOCATION,
        connection=CONNECTION_NAME,
        entity_operations={
            "incident": ["LIST", "GET", "CREATE", "UPDATE"],
            "kb_knowledge": ["LIST", "GET"],
            "sys_user": ["LIST", "GET"],
            "change_request": ["LIST", "GET", "CREATE", "UPDATE"],
            "problem": ["LIST", "GET", "CREATE", "UPDATE"],
            "sc_request": ["LIST", "GET", "CREATE"],
        },
        tool_name_prefix="servicenow",
    )
    return servicenow_toolset

async def get_tools_from_toolset(toolset):
    return await toolset.get_tools()

# 4️⃣ Process agent responses
def process_agent_response(runner, session_id, query):
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=query)])
    events = runner.run(session_id=session_id, user_id="user-123", new_message=content)
    response_parts = []
    for event in events:
        if event.content:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    response_parts.append(part.text)
                # You can add more detailed output for debugging here if needed
    return "\n".join(response_parts) or "No text response received."

# 5️⃣ Interactive CLI
def start_interactive_chat(runner, session_id):
    print("\n" + "=" * 60)
    print("🤖 ServiceNow Agent Chat - Ready!")
    print("=" * 60)
    print("Type 'quit' to exit.")
    print("-" * 60)
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Thanks for using the local runner. Goodbye!")
                break
            if not user_input:
                continue

            response = process_agent_response(runner, session_id, user_input)
            print(f"\nAgent: {response}\n" + "-" * 50)
        except Exception as e:
            print(f"\nAn error occurred: {e}")

# 6️⃣ Main initialization function
async def initialize_agent():
    toolset = create_servicenow_toolset()
    tools = await get_tools_from_toolset(toolset)

    agent = LlmAgent(
        name="servicenow_agent",
        model=GEMINI_MODEL,
        instruction=(
            "You are a ServiceNow assistant. Use your tools to fulfill user requests. "
            "Be concise and helpful. "
            "Important Rule 1: When listing items like incidents, do not use any sorting parameters "
            "as the backend does not support it. If a user asks for 'recent' or 'latest' items, "
            "perform a standard list operation and inform the user that the results are not sorted."
            "Important Rule 2: After you successfully create an entity and get an ID back, "
            "if an immediate attempt to retrieve that same entity fails with a 'Not Found' error, "
            "do not assume the creation failed. Instead, inform the user that the system "
            "might need a moment to process the new record and suggest they try retrieving it again in a minute."
        ),
        tools=tools,
    )

    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    session = await session_service.create_session(app_name="servicenow_agent", user_id="user-123")

    runner = Runner(
        app_name="servicenow_agent",
        agent=agent,
        session_service=session_service,
        artifact_service=artifact_service,
    )
    return runner, session.id

if __name__ == "__main__":
    try:
        runner, session_id = asyncio.run(initialize_agent())
        start_interactive_chat(runner, session_id)
    except Exception as e:
        print(f"\n💥 A critical error occurred during startup: {e}")
        sys.exit(1)