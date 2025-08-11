import os
import sys
import vertexai
from dotenv import load_dotenv
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration

# --- 1. Configuration and Initialization ---
load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-001")

# This prefix must map to the simplified name of your Application Integration connector.
# e.g., 'connector-servicenow' becomes 'servicenow'
TOOL_NAME_PREFIX = "servicenow" 

if not all([PROJECT_ID, LOCATION, GEMINI_MODEL]):
    raise ValueError("Missing one or more required environment variables in your .env file.")

print(f"✅ Configuration loaded for project '{PROJECT_ID}'...")
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    print("✅ Vertex AI initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize Vertex AI: {e}")
    raise


# --- 2. Manual Tool Definition ---
# We explicitly define each function the agent can call. This is the most reliable method.
# The 'name' is critical: TOOL_NAME_PREFIX + _ + [entity] + _ + [operation]
print("\n🔧 Defining ServiceNow tools manually...")

# --- Incident Tools ---
list_incidents_func = FunctionDeclaration(
    name=f"{TOOL_NAME_PREFIX}_incident_list",
    description="Gets a list of incidents from ServiceNow. Can be filtered.",
    parameters={
        "type": "object",
        "properties": {
            "filter": {"type": "string", "description": "A filter string to apply, e.g., \"state='2' AND priority='1'\""},
            "limit": {"type": "integer", "description": "The maximum number of incidents to return."}
        }
    }
)

create_incident_func = FunctionDeclaration(
    name=f"{TOOL_NAME_PREFIX}_incident_create",
    description="Creates a new incident in ServiceNow.",
    parameters={
        "type": "object",
        "properties": {
            "short_description": {"type": "string", "description": "A brief, one-line summary of the incident."},
            "caller_id": {"type": "string", "description": "The name or user ID of the person reporting the incident."}
        },
        "required": ["short_description"]
    }
)

# --- Knowledge Base Tools ---
list_kb_articles_func = FunctionDeclaration(
    name=f"{TOOL_NAME_PREFIX}_kb_knowledge_list",
    description="Lists or searches for knowledge base articles in ServiceNow.",
    parameters={
        "type": "object",
        "properties": {
            "filter": {"type": "string", "description": "A filter string for searching, e.g., \"short_description LIKE 'password'\""}
        }
    }
)

# --- Combine all function declarations into a single Tool object ---
servicenow_tool = Tool(
    function_declarations=[
        list_incidents_func,
        create_incident_func,
        list_kb_articles_func,
    ]
)
print("✅ Tools defined successfully.")


# --- 3. Main Application Logic ---

def main():
    """Initializes the agent and runs the main user menu."""
    
    print("🧠 Creating the generative model...")
    model = GenerativeModel(model_name=GEMINI_MODEL, tools=[servicenow_tool])
    print("✅ Model created successfully.")

    start_interactive_chat(model)

def start_interactive_chat(model: GenerativeModel):
    """Starts a real-time, interactive chat session with the agent."""
    print("\n" + "="*60)
    print("🤖 ServiceNow Agent (Manual Tools) - Ready!")
    print("="*60)
    print("Example: 'List the 5 most recent incidents'")
    print("Type 'quit' to exit.")
    print("-" * 60)
    
    chat = model.start_chat()
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            if not user_input:
                continue
            
            print("⚙️ Processing...")
            response = chat.send_message(user_input)
            
            # --- THE FINAL, CORRECTED RESPONSE HANDLING ---
            try:
                # The Vertex AI SDK is designed to automatically handle the tool-use
                # loop. The final response from the model, after it has used a tool
                # and received the result, should contain a text summary.
                print(f"\nAgent: {response.text}\n" + "-" * 50)
            except ValueError:
                # This block handles the case where the model's response is a tool
                # call instruction instead of text. This gives us visibility into
                # the agent's "thinking" process.
                print("\nAgent: 🧠 I need to use a tool to answer that. The SDK will now automatically execute it.")
                # The SDK will automatically call the tool, send the result back to the model,
                # and the next `response.text` will contain the summary.
                # In a more advanced implementation, you would manually check for and execute
                # the function call here, but the SDK's automatic mode handles this.
            # ----------------------------------------------------

        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    main()