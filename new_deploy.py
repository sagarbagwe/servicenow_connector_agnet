import os
import asyncio
from dotenv import load_dotenv
import vertexai
from vertexai import agent_engines
from vertexai.preview import reasoning_engines
from google.adk.agents import LlmAgent
from google.adk.tools.application_integration_tool.application_integration_toolset import ApplicationIntegrationToolset

# -----------------------------------------------------------
# Load environment variables from .env
# -----------------------------------------------------------
load_dotenv()

# Local values (for local run only)
PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
STAGING_BUCKET = os.environ["STAGING_BUCKET"]
CONNECTION_NAME = os.environ["CONNECTION_NAME"]
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
api_key = os.environ.get("api_key")


# -----------------------------------------------------------
# Build agent - Async function
# -----------------------------------------------------------
async def build_agent():
    toolset = ApplicationIntegrationToolset(
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

    instruction = (
        "You are a ServiceNow assistant. Use your tools to fulfill user requests. "
        "Be concise and helpful.\n\n"
        "Rule 1: When listing items (e.g. incidents), do not use sorting parameters "
        "as the backend does not support them. If a user asks for 'recent' or 'latest', "
        "run a normal list and clarify that results are unsorted.\n\n"
        "Rule 2: After creating an entity, if fetching it immediately fails with 'Not Found', "
        "explain that ServiceNow may take a moment to index the record and suggest retrying."
    )

    return LlmAgent(
        name="servicenow_agent",
        model=GEMINI_MODEL,
        instruction=instruction,
        tools=await toolset.get_tools(),
    )


# -----------------------------------------------------------
# Wrap into AdkApp - Async
# -----------------------------------------------------------
async def build_app():
    agent = await build_agent()
    return reasoning_engines.AdkApp(agent=agent, enable_tracing=True)


# -----------------------------------------------------------
# Deploy to Agent Engine - Async
# -----------------------------------------------------------
async def deploy():
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        print("🔑 Using API key authentication")

    vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)

    app = await build_app()

    remote_app = agent_engines.create(
        agent_engine=app,
        requirements=[
            "google-cloud-aiplatform[adk,agent_engines]>=1.88.0",
            "google-adk>=0.4.0",
            "google-genai>=0.2.0",
            "python-dotenv>=1.0.0",
            "packaging>=24.0",
            "google-auth>=2.30.0",
            "pydantic==2.11.7",
            "cloudpickle==3.1.1",
        ],
        # ✅ Reserved vars removed, using custom names
        env_vars={
            "APP_PROJECT_ID": PROJECT_ID,
            "APP_LOCATION": LOCATION,
            "APP_CONNECTION_NAME": CONNECTION_NAME,
            "APP_GEMINI_MODEL": GEMINI_MODEL,
            "APP_API_KEY": api_key or "",
        },
        display_name="servicenow-agent",
    )

    print("✅ Deployed successfully!")
    print("Resource name:", remote_app.resource_name)
    return remote_app


# -----------------------------------------------------------
# Remote interactive CLI
# -----------------------------------------------------------
async def start_remote_chat(remote_app):
    print("\n🤖 Remote ServiceNow Agent (Agent Engine) - Ready!")
    session = remote_app.create_session(user_id="user-123")

    while True:
        user_input = input("\n🧑 You: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye!")
            break

        print("\n[DEBUG] Sending query to Agent Engine...")
        for event in remote_app.stream_query(
            user_id="user-123",
            session_id=session["id"],
            message=user_input,
        ):
            # Always print the raw event for debugging
            print("\n[EVENT]", event)

            # If it's text output
            if "parts" in event and event["parts"]:
                for part in event["parts"]:
                    if "text" in part:
                        print(part["text"], end="", flush=True)

        print()  # new line after response
    print("\n🤖 Remote ServiceNow Agent (Agent Engine) - Ready!")
    session = remote_app.create_session(user_id="user-123")

    while True:
        user_input = input("\n🧑 You: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye!")
            break

        for event in remote_app.stream_query(
            user_id="user-123",
            session_id=session["id"],
            message=user_input,
        ):
            if "parts" in event and event["parts"]:
                for part in event["parts"]:
                    if "text" in part:
                        print(part["text"], end="", flush=True)
        print()


# -----------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------
async def main():
    remote_app = await deploy()
    await start_remote_chat(remote_app)


if __name__ == "__main__":
    try:
        api_key = os.environ["api_key"]
    except KeyError:
        api_key = None
        print("⚠️ Warning: GOOGLE_API_KEY not found in environment variables.")
    asyncio.run(main())
