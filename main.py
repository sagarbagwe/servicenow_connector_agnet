# deploy_servicenow_agent.py

import asyncio
import vertexai
from vertexai.preview import reasoning_engines
from google.adk.agents import LlmAgent
from google.adk.tools.application_integration_tool.application_integration_toolset import ApplicationIntegrationToolset

PROJECT_ID = "sadproject2025"
LOCATION = "us-central1"
STAGING_BUCKET = "gs://vertex-adk-agent-bucket-20250812160823"
GEMINI_MODEL = "gemini-2.5-pro"
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

# --- 2. Build a deployable wrapper ---
class DeployableAgent:
    def __init__(self, agent):
        self.agent = agent

    async def query(self, input_text: str):
        # Matches ReasoningEngine's expected callable
        return await self.agent.query(input_text)

async def build_deployable_agent():
    toolset = create_servicenow_toolset()
    tools = await toolset.get_tools()

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

    return DeployableAgent(agent)

# --- 3. Deployment ---
def main():
    vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)
    print("\n🚀 Deploying ServiceNow Agent to Vertex AI Agent Engine...")

    deployable_agent = asyncio.run(build_deployable_agent())

    try:
        remote_app = reasoning_engines.ReasoningEngine.create(
            reasoning_engine=deployable_agent,
            requirements=[
                "google-cloud-aiplatform[reasoning_engines]>=1.34.0",
                "google-adk>=0.1.0",
            ],
            display_name="servicenow-production-agent-hardcoded",
            description="ServiceNow Assistant Agent with hardcoded config",
        )

        print(f"✅ Agent deployed successfully!")
        print(f"Resource Name: {remote_app.resource_name}")
        print(f"Agent ID: {remote_app.name}")
        return remote_app

    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
