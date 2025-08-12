import os
import sys
import asyncio
from dotenv import load_dotenv
import vertexai
from packaging import version
from importlib.metadata import version as pkg_version
import google.auth
from google.auth.transport.requests import Request

from google.adk.agents import LlmAgent
from google.adk.tools.application_integration_tool.application_integration_toolset import ApplicationIntegrationToolset
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.genai import types as genai_types

# -----------------------------------------------------------
# .env file example:
# GOOGLE_CLOUD_PROJECT=your-project-id
# GOOGLE_CLOUD_LOCATION=us-central1
# CONNECTION_NAME=sn-connector-prod
# GEMINI_MODEL=gemini-1.5-pro
# -----------------------------------------------------------

# 1️⃣ Load environment variables
load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
CONNECTION_NAME = os.getenv("CONNECTION_NAME")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

if not all([PROJECT_ID, LOCATION, CONNECTION_NAME, GEMINI_MODEL]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

print(f"✅ Configuration loaded for project '{PROJECT_ID}' (Location: {LOCATION})")

# 2️⃣ Configure Authentication via Vertex AI
# This is the only required path when using Application Integration tools.
print("🔑 Proceeding with default Vertex AI authentication...")
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    print("✅ Vertex AI initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize Vertex AI: {e}")
    sys.exit(1)

print("🔑 Authenticating application with default credentials...")
try:
    credentials, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())
    print("✅ Application successfully authenticated.")
except Exception as e:
    print(f"❌ Failed to authenticate application: {e}")
    print("Please run 'gcloud auth application-default login' in your terminal and try again.")
    sys.exit(1)


# 3️⃣ Detect google-adk version
try:
    adk_version = pkg_version("google-adk")
except Exception:
    adk_version = "0.0.0"

print(f"📦 Detected google-adk version: {adk_version}")

# -----------------------------------------------------------
# Create ServiceNow toolset
# -----------------------------------------------------------
def create_servicenow_toolset():
    print("\n🛠️ Creating ServiceNow Application Integration Toolset...")
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
    print("✅ ServiceNow toolset created successfully")
    return servicenow_toolset


# -----------------------------------------------------------
# Load tools from the toolset
# -----------------------------------------------------------
async def get_tools_from_toolset(toolset):
    print("⏳ Loading tools from ServiceNow toolset...")
    tools = await toolset.get_tools()
    print(f"✅ Successfully loaded {len(tools)} tools.")
    return tools


# -----------------------------------------------------------
# Process agent responses
# -----------------------------------------------------------
def process_agent_response(runner, session_id, query):
    print(f"\n🧠 Processing: '{query}'...")
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=query)])
    events = runner.run(session_id=session_id, user_id="user-123", new_message=content)

    response_parts = []
    for event in events:
        if event.content:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    response_parts.append(part.text)
                elif getattr(part, "function_call", None):
                    fn = part.function_call
                    response_parts.append(f"🔧 Function call: {fn.name} with args: {repr(fn.args)}")
                elif getattr(part, "function_response", None):
                    fr = part.function_response
                    response_parts.append(f"📦 Function response from {fr.name}: {fr.response}")

    return "\n".join(response_parts) or "No text response received from the agent."


# -----------------------------------------------------------
# Interactive CLI
# -----------------------------------------------------------
def start_interactive_chat(runner, session_id):
    print("\n" + "=" * 60)
    print("🤖 ServiceNow Agent Chat - Ready!")
    print("=" * 60)
    print("Example: 'Show me the 5 most recent incidents'")
    print("Type 'quit' to exit.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Thanks for using ServiceNow Agent. Goodbye!")
                break
            if not user_input:
                continue

            response = process_agent_response(runner, session_id, user_input)
            print(f"\nAgent: {response}\n" + "-" * 50)
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred during processing: {e}")


# -----------------------------------------------------------
# Main initialization
# -----------------------------------------------------------
async def initialize_agent():
    toolset = create_servicenow_toolset()
    tools = await get_tools_from_toolset(toolset)

    print("⚙️ Creating the LlmAgent...")

    # The agent's model parameter should be a string.
    # The authentication is handled globally at the start of the script.
    agent_kwargs = {
        "name": "servicenow_agent",
        "model": GEMINI_MODEL,
        "instruction": (
            "You are a ServiceNow assistant. Use your tools to fulfill user requests "
            "related to incidents, knowledge articles, and users. Be concise and helpful."
        ),
        "tools": tools,
    }

    agent = LlmAgent(**agent_kwargs)

    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    session = await session_service.create_session(app_name="servicenow_agent", user_id="user-123")

    runner = Runner(
        app_name="servicenow_agent",
        agent=agent,
        session_service=session_service,
        artifact_service=artifact_service,
    )

    print("✅ ServiceNow agent initialized successfully")
    return runner, session.id


# -----------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------
def main():
    print("\n🚀 ServiceNow Agent (Google ADK) Starting...")
    try:
        runner, session_id = asyncio.run(initialize_agent())
        start_interactive_chat(runner, session_id)
    except Exception as e:
        print(f"\n💥 A critical error occurred during startup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
