import os
import sys
import asyncio
from dotenv import load_dotenv

import vertexai
from vertexai.preview import reasoning_engines
import google.auth
from google.auth.transport.requests import Request

# ADK components from your working local_runner.py
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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro-001")

if not all([PROJECT_ID, LOCATION, CONNECTION_NAME, GEMINI_MODEL]):
    raise ValueError("Missing required environment variables. Please check your .env file.")


# 2️⃣ Define a simpler wrapper class.
# Its job is only to hold the pre-initialized runner and provide the query method.
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

        try:
            events = self.runner.run(session_id=session_id, user_id="user-123", new_message=content)
            response_parts = [
                part.text
                for event in events
                if event.content
                for part in event.content.parts
                if getattr(part, "text", None)
            ]
            final_response = "\n".join(response_parts) or "No text response received."
            return {"response": final_response}
        except Exception as e:
            print(f"ERROR during agent execution: {e}")
            return {"error": str(e)}


# 3️⃣ Use an async factory to correctly initialize the agent and tools.
async def create_deployable_app() -> DeployableADKAgent:
    """
    Asynchronously creates and initializes all ADK components
    before passing them to the synchronous wrapper class.
    """
    print("🔧 Initializing agent components asynchronously...")
    
    # Create the toolset
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

    # Asynchronously resolve the tools, just like in local_runner.py
    resolved_tools = await servicenow_toolset.get_tools()
    print(f"🛠️  {len(resolved_tools)} tools resolved.")

    # Create the agent with the fully resolved tools
    agent = LlmAgent(
        name="servicenow_agent",
        model=GEMINI_MODEL,
        instruction="You are a ServiceNow assistant. Use your tools to fulfill user requests.",
        tools=resolved_tools,
    )
    
    # Create the other services and the runner
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    runner = Runner(
        app_name="servicenow_agent",
        agent=agent,
        session_service=session_service,
        artifact_service=artifact_service,
    )
    print("🏃 Runner is configured.")

    # Finally, create the deployable instance with the ready-to-use runner
    return DeployableADKAgent(runner=runner)


# 4️⃣ Main deployment function is now async
async def main():
    # Await the creation of our fully initialized deployable app
    deployable_app = await create_deployable_app()

    STAGING_BUCKET = f"gs://{PROJECT_ID}-adk-staging-bucket"
    vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)
    
    print("\n🚀 Deploying the ServiceNow agent to Vertex AI Reasoning Engine...")
    print("This may take several minutes...")
    
    try:
        remote_app = reasoning_engines.ReasoningEngine.create(
            reasoning_engine=deployable_app,
            requirements=[
                "google-cloud-aiplatform[reasoning_engines]>=1.34.0",
                "google-adk>=0.1.0",
                "google-generativeai",
                "python-dotenv"
            ],
            display_name="servicenow-production-agent-final",
            description="ServiceNow Assistant Agent deployed via a custom async wrapper.",
            
        )
        
        print(f"\n✅ Agent deployed successfully!")
        print(f"   Resource Name: {remote_app.resource_name}")
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        # import traceback
        # traceback.print_exc()
        sys.exit(1)


# 5️⃣ Use asyncio.run() to start the main async function
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"A critical error occurred: {e}")
        sys.exit(1)