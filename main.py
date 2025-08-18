import asyncio
import vertexai
from vertexai.preview import reasoning_engines

# Imports for the agent and tools
from google.adk.agents import LlmAgent
from google.adk.tools.application_integration_tool.application_integration_toolset import ApplicationIntegrationToolset

# Imports needed for the corrected wrapper
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.generativeai import types as genai_types

PROJECT_ID = "sadproject2025"
LOCATION = "us-central1"
STAGING_BUCKET = "gs://vertex-adk-agent-bucket-20250812160823"
GEMINI_MODEL = "gemini-1.5-pro"
CONNECTION_NAME = "sn-connector-prod"

# --- 1. Create ServiceNow Toolset ---
def create_servicenow_toolset():
    return ApplicationIntegrationToolset(
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

# --- 2. Build a Corrected Deployable Wrapper ---
class DeployableAgent:
    def __init__(self, agent):
        self.agent = agent
        # Initialize the services and runner once
        self.session_service = InMemorySessionService()
        self.artifact_service = InMemoryArtifactService()
        self.runner = Runner(
            app_name="servicenow-agent-deployed",
            agent=self.agent,
            session_service=self.session_service,
            artifact_service=self.artifact_service,
        )

    async def query(self, input_text: str):
        """
        This method now correctly uses the Runner to process the input.
        """
        # Create a new session for each query to keep it stateless
        session = await self.session_service.create_session(app_name="servicenow-agent-deployed")
        message = genai_types.Content(parts=[genai_types.Part(text=input_text)])
        
        response_parts = []
        # The runner.run() method yields events; we collect the text parts
        for event in self.runner.run(session_id=session.id, new_message=message):
            if event.content:
                for part in event.content.parts:
                    if part.text:
                        response_parts.append(part.text)
        
        # Join all text parts into a single response string
        return "".join(response_parts)

async def build_deployable_agent():
    toolset = create_servicenow_toolset()
    agent = LlmAgent(
        name="servicenow_agent",
        model=GEMINI_MODEL,
        instruction=(
            "You are a ServiceNow assistant. Use your tools to fulfill user requests. "
            "Be concise and helpful. "
            "Important Rule 1: When listing items like incidents, do not use any sorting parameters "
            "as the backend does not support it. If a user asks for 'recent' or 'latest' items, "
            "perform a standard list operation and inform the user that the results are not sorted."
        ),
        tools=[toolset],
    )
    return DeployableAgent(agent)

# --- 3. Deployment ---
def main():
    vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)
    print("\n🚀 Deploying ServiceNow Agent with corrected code...")

    deployable_agent = asyncio.run(build_deployable_agent())

    try:
        remote_app = reasoning_engines.ReasoningEngine.create(
            reasoning_engine=deployable_agent,
            requirements=[
                "google-cloud-aiplatform[reasoning_engines]>=1.34.0",
                "google-adk>=0.1.0",
                "google-generativeai", # ❗️ This line is added to fix the error
            ],
            display_name="servicenow-agent-fixed-v2",
            description="ServiceNow Assistant Agent with corrected runner logic and dependencies",
        )
        print(f"✅ Agent deployed successfully!")
        print(f"Resource Name: {remote_app.resource_name}")
    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()