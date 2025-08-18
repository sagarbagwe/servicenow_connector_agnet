import vertexai
from vertexai import agent_engines
import os

# --- 1. Configuration ---
PROJECT_ID = "sadproject2025"
LOCATION = "us-central1"

# The unique resource name of your deployed agent
REASONING_ENGINE_ID = "projects/113207776071/locations/us-central1/reasoningEngines/2262346329218875392"

# --- 2. Initialize and Connect to the Agent ---
print(f"🚀 Initializing Vertex AI for project '{PROJECT_ID}'...")
vertexai.init(project=PROJECT_ID, location=LOCATION)

print(f"🔎 Retrieving deployed agent: {REASONING_ENGINE_ID}")
agent = agent_engines.get(REASONING_ENGINE_ID)
print("✅ Agent retrieved successfully.\n")

# --- 3. Query the Agent ---
prompt = "Show me the 5 most recent incidents"
print(f"💬 Sending prompt to agent: '{prompt}'")
print("-" * 30)

try:
    # ❗️ FIX: Call the .query() method, which matches your deployed code
    response = agent.query(input_text=prompt)

    print("🤖 Agent Response:")
    print(response)

except Exception as e:
    print(f"\n❌ An error occurred during the query: {e}")

print("-" * 30)
print("✅ Query finished.")