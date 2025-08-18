import asyncio
import os
import sys
from dotenv import load_dotenv
import vertexai
from vertexai.preview import reasoning_engines
from google.adk.agents import LlmAgent
from google.adk.tools.application_integration_tool.application_integration_toolset import ApplicationIntegrationToolset
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.genai import types as genai_types

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
CONNECTION_NAME = os.getenv("CONNECTION_NAME")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro-001")
STAGING_BUCKET = f"gs://{PROJECT_ID}-adk-staging-bucket"

if not all([PROJECT_ID, LOCATION, CONNECTION_NAME, GEMINI_MODEL]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

# ---------- Lightweight Deployable Wrapper ----------
class DeployableADKAgent:
    def __init__(self, runner: Runner):
        self.runner = runner
        print("✅ Deployable Agent wrapper is ready.")

    def query(self, **kwargs) -> dict:
        prompt = kwargs.get("text", "")
        if not prompt:
            return {"error": "Input 'text' cannot be empty."}
        session_id = f"session-{os.urandom(8).hex()}"
        content = genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])
        events = self.runner.run(session_id=session_id, user_id="user-123", new_message=content)
        parts = [p.text for e in events if e.content for p in e.content.parts if getattr(p, "text", None)]
        return {"response": "\n".join(parts) or "No response received."}

# ---------- Initialize Agent & Runner ----------
async def create_deployable_app():
    print("🔧 Initializing agent components asynchronously...")

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

    resolved_tools = await toolset.get_tools()
    print(f"🛠️  {len(resolved_tools)} tools resolved.")

    agent = LlmAgent(
        name="servicenow_agent",
        model=GEMINI_MODEL,
        instruction="You are a ServiceNow assistant. Use your tools to fulfill user requests.",
        tools=resolved_tools,
    )

    runner = Runner(
        app_name="servicenow_agent",
        agent=agent,
        session_service=InMemorySessionService(),
        artifact_service=InMemoryArtifactService(),
    )

    print("🏃 Runner is configured.")
    return DeployableADKAgent(runner=runner)

# ---------- Deploy to Agent Engine ----------
async def main():
    vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)

    deployable_app = await create_deployable_app()

    print("\n🚀 Deploying the ServiceNow agent to Vertex AI Reasoning Engine...")
    try:
        remote_app = reasoning_engines.ReasoningEngine.create(
            reasoning_engine=deployable_app,
            requirements=[
                "google-cloud-aiplatform[reasoning_engines]>=1.34.0",
                "google-adk>=0.1.0",
                "google-generativeai",
                "python-dotenv"
            ],
            display_name="servicenow-production-agent",
            description="ServiceNow Assistant Agent deployed via ADK",
        )

        print(f"✅ Agent Engine deployed successfully!")
        print(f"   Resource Name: {remote_app.resource_name}")
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"A critical error occurred: {e}")
        sys.exit(1)
