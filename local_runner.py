import os
import sys
import vertexai
from dotenv import load_dotenv

# Imports for the new Agent Engine deployment method
from vertexai.preview.reasoning_engines import AdkApp
from vertexai import agent_engines

# Imports from your original script to define the agent
from google.adk.agents import LlmAgent
from google.adk.tools.application_integration_tool.application_integration_toolset import ApplicationIntegrationToolset

# --- 1. Configuration ---
print("🚀 Starting ServiceNow Agent deployment process...")
load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
CONNECTION_NAME = os.getenv("CONNECTION_NAME")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

# A GCS bucket is required for staging the agent artifacts during deployment.
# Ensure this bucket exists in your project.
STAGING_BUCKET = f"gs://{PROJECT_ID}-agent-engine-staging" 

if not all([PROJECT_ID, LOCATION, CONNECTION_NAME, GEMINI_MODEL]):
    print("❌ Error: Missing required environment variables. Please check your .env file.", file=sys.stderr)
    sys.exit(1)

print(f"✅ Configuration loaded for project '{PROJECT_ID}'")

# Initialize Vertex AI with the project, location, and staging bucket.
vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)

# --- 2. Define Your Agent ---
# This section defines the tools and instructions for your ServiceNow agent.
print("🛠️  Defining the ServiceNow Agent...")

# Create the ServiceNow toolset using Application Integration
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

# Define the LlmAgent that will use the toolset
servicenow_agent = LlmAgent(
    name="servicenow_agent",
    model=GEMINI_MODEL,
    instruction=(
        "You are a ServiceNow assistant. Use your tools to fulfill user requests. "
        "Be concise and helpful. "
        "Important Rule 1: When listing items like incidents, do not use any sorting parameters "
        "as the backend does not support it. If a user asks for 'recent' or 'latest' items, "
        "perform a standard list operation and inform the user that the results are not sorted."
    ),
    tools=[servicenow_toolset], # Pass the toolset directly to the agent
)
print("✅ Agent definition complete.")

# --- 3. Package and Deploy ---
# This section packages the agent and deploys it to Agent Engine.
print("📦 Packaging agent with AdkApp...")
app = AdkApp(
    agent=servicenow_agent,
    enable_tracing=True,
)

# Explicitly define all necessary Python packages for the deployment environment.
# This prevents dependency errors during startup.
deployment_requirements = [
    "google-cloud-aiplatform[agent_engines,adk]>=1.55.0",
    "google-adk>=0.1.0",
    "python-dotenv>=1.0.0",
    "google-auth>=2.29.0",
]

print("🚢 Deploying to Vertex AI Agent Engine with explicit requirements... (This may take 15-20 minutes)")

# The create() function handles packaging, uploading, and deployment.
# We pass the requirements list directly to ensure a stable environment.
remote_app = agent_engines.create(
    app,
    requirements=deployment_requirements
)

print("\n🎉 Deployment successful!")
print("Agent Resource Name:", remote_app.resource_name)
print(remote_app)