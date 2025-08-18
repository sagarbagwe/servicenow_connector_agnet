import os
import sys
import asyncio
import streamlit as st
from dotenv import load_dotenv
import vertexai
from packaging import version
from importlib.metadata import version as pkg_version
import google.auth
from google.auth.transport.requests import Request
import json
from collections.abc import Generator

from google.adk.agents import LlmAgent
from google.adk.tools.application_integration_tool.application_integration_toolset import ApplicationIntegrationToolset
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.genai import types as genai_types

# -----------------------------------------------------------
# App Configuration & Styling
# -----------------------------------------------------------
st.set_page_config(
    page_title="ServiceNow AI Agent",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

# Global CSS for polished UI
st.markdown(
    """
<style>
/* Base font & smoothing */
html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
* { -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }

/* Spinner accent */
.stSpinner > div > div { border-top-color: #6366f1; }

/* Buttons */
.stButton>button { border-radius: 0.75rem; padding: 0.5rem 0.9rem; }

/* Chat bubbles */
.chat-bubble { padding: 0.85rem 1rem; border-radius: 1rem; margin: 0.35rem 0; max-width: 82%; word-wrap: break-word; box-shadow: 0 2px 6px rgba(0,0,0,0.06); border: 1px solid rgba(0,0,0,0.05); }
.chat-user { background-color: #DCF8C6; margin-left: auto; }
.chat-assistant { background-color: #F5F5F5; margin-right: auto; }

/* Final response emphasis */
.final-response { background: linear-gradient(90deg, #EEF2FF, #E0E7FF); border: 1px solid #C7D2FE; border-radius: 1rem; padding: 1rem; margin-top: 0.5rem; box-shadow: 0 3px 8px rgba(99,102,241,0.08); }
.final-response h4 { margin: 0 0 .5rem 0; font-weight: 600; }

/* Tool activity card */
.tool-card { background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 0.75rem; padding: 0.75rem; margin: 0.5rem 0; font-size: 0.92rem; overflow-x: auto; }
.tool-card .title { font-weight: 600; margin-bottom: 0.35rem; }

/* Expander tweaks */
.block-container .st-emotion-cache-ue6h4q p, .block-container p { line-height: 1.55; }

/* Code fences */
pre, code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 0.90rem; }

/* Header badge */
.header-badge { display: inline-flex; align-items: center; gap: .45rem; background: #EEF2FF; color: #3730A3; padding: .35rem .6rem; border-radius: 999px; border: 1px solid #C7D2FE; font-size: .80rem; }

/* Tiny muted text */
.muted { color: #6B7280; font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
<h1 style="margin-bottom: .25rem;">🤖 ServiceNow AI Agent</h1>
<span class="muted">Powered by Google Agent Development Kit (ADK) and Vertex AI</span>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------
# Load Environment & Agent Instructions
# -----------------------------------------------------------
load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
CONNECTION_NAME = os.getenv("CONNECTION_NAME")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

AGENT_INSTRUCTIONS = (
    "You are a ServiceNow assistant. Use your tools to fulfill user requests. "
    "Be concise and helpful. "
    "Important Rule 1: When listing items like incidents, do not use any sorting parameters "
    "as the backend does not support it. If a user asks for 'recent' or 'latest' items, "
    "perform a standard list operation and inform the user that the results are not sorted. "
    "Important Rule 2: After you successfully create an entity and get an ID back, "
    "if an immediate attempt to retrieve that same entity fails with a 'Not Found' error, "
    "do not assume the creation failed. Instead, inform the user that the system "
    "might need a moment to process the new record and suggest they try retrieving it again in a minute."
)

# -----------------------------------------------------------
# Sidebar for Configuration and Controls
# -----------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info(
        f"**Project ID:** `{PROJECT_ID}`\n\n"
        f"**Location:** `{LOCATION}`\n\n"
        f"**Connection:** `{CONNECTION_NAME}`\n\n"
        f"**Model:** `{GEMINI_MODEL}`"
    )

    with st.expander("🧠 View Agent Instructions"):
        st.markdown(f"_{AGENT_INSTRUCTIONS}_")

    st.header("✅ Status")
    try:
        adk_version = pkg_version("google-adk")
        st.success(f"google-adk v{adk_version}")
    except Exception:
        st.warning("`google-adk` not found.")
        st.stop()

    st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# -----------------------------------------------------------
# Agent Initialization (Cached for performance)
# -----------------------------------------------------------
@st.cache_resource
def initialize_agent():
    """
    Initialize Vertex AI, authenticate, create the ServiceNow toolset,
    and instantiate the agent and runner.
    """
    if not all([PROJECT_ID, LOCATION, CONNECTION_NAME, GEMINI_MODEL]):
        st.error("Missing required environment variables. Please check your `.env` file.")
        st.stop()

    # 1️⃣ Configure Authentication via Vertex AI
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        credentials, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"]) 
        credentials.refresh(Request())
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        st.warning("Please run `gcloud auth application-default login` in your terminal and try again.")
        st.stop()

    # 2️⃣ Create ServiceNow Toolset
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

    # 3️⃣ Load tools and create agent
    try:
        tools = asyncio.run(servicenow_toolset.get_tools())
    except Exception as e:
        st.error(f"Failed to load tools from Application Integration: {e}")
        st.stop()

    agent = LlmAgent(
        name="servicenow_agent",
        model=GEMINI_MODEL,
        instruction=AGENT_INSTRUCTIONS,
        tools=tools,
    )

    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    session = asyncio.run(session_service.create_session(app_name="servicenow_agent", user_id="streamlit-user"))

    runner = Runner(
        app_name="servicenow_agent",
        agent=agent,
        session_service=session_service,
        artifact_service=artifact_service,
    )
    return runner, session.id

# -----------------------------------------------------------
# Function to stream agent response for a better UX
# -----------------------------------------------------------
def stream_agent_response(runner: Runner, session_id: str, query: str) -> Generator[dict, None, None]:
    """
    Sends the user's query to the agent and yields structured events
    for tool calls, tool responses, and the final text response.
    Formats list-based tool responses as bullet lists instead of raw JSON.
    """
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=query)])
    events = runner.run(session_id=session_id, user_id="streamlit-user", new_message=content)

    final_response_parts = []
    for event in events:
        if not event.content:
            continue
        for part in event.content.parts:
            if fn_call := getattr(part, "function_call", None):
                args_str = json.dumps(dict(fn_call.args), indent=2)
                yield {
                    "type": "tool_call",
                    "content": f"<div class='tool-card'><div class='title'>🛠️ Tool Call</div><code>{fn_call.name}</code>\n\n```json\n{args_str}\n```</div>",
                }
            elif fn_response := getattr(part, "function_response", None):
                response_str = ""
                try:
                    response_data = json.loads(fn_response.response)
                    if isinstance(response_data, list) and response_data:
                        list_lines = []
                        for item in response_data:
                            if isinstance(item, dict):
                                item_summary = ", ".join(f"`{k}`: {v}" for k, v in item.items() if v)
                            else:
                                item_summary = str(item)
                            list_lines.append(f"- {item_summary}")
                        response_str = "\n".join(list_lines)
                    else:
                        response_str = f"```json\n{json.dumps(response_data, indent=2)}\n```"
                except (json.JSONDecodeError, TypeError):
                    response_str = str(fn_response.response)
                yield {
                    "type": "tool_response",
                    "content": f"<div class='tool-card'><div class='title'>📩 Tool Response from <code>{fn_response.name}</code></div>{response_str}</div>",
                }
            elif text := getattr(part, "text", None):
                final_response_parts.append(text)

    if final_response_parts:
        full_text = "".join(final_response_parts)
        yield {"type": "final_text", "content": full_text}

# -----------------------------------------------------------
# Main Chat Application Logic
# -----------------------------------------------------------
with st.spinner("🚀 Initializing ServiceNow Agent..."):
    try:
        runner, session_id = initialize_agent()
    except Exception as e:
        st.error("A critical error occurred during agent initialization.")
        st.exception(e)
        st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! How can I help you with ServiceNow today?\n\n**For example:** `List the 5 most recent incidents`"
    }]

# Render a message in a bubble
def render_message(role: str, content: str):
    bubble_class = "chat-assistant" if role == "assistant" else "chat-user"
    with st.chat_message(role):
        st.markdown(f"<div class='chat-bubble {bubble_class}'>{content}</div>", unsafe_allow_html=True)

# Display history
for message in st.session_state.messages:
    render_message(message["role"], message["content"])

# Handle input
if prompt := st.chat_input("What would you like to do?"):
    # Show user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message("user", prompt)

    # Assistant container
    with st.chat_message("assistant"):
        tool_activity_md = []
        final_response = ""

        # Collapsible agent activity
        activity_expander = st.expander("🔎 Show Agent Activity")
        activity_container = activity_expander.container()
        response_placeholder = st.empty()

        with st.spinner("🤖 The agent is thinking..."):
            for event in stream_agent_response(runner, session_id, prompt):
                if event["type"] in ["tool_call", "tool_response"]:
                    tool_activity_md.append(event["content"])  # already wrapped in .tool-card
                    activity_container.markdown("\n".join(tool_activity_md), unsafe_allow_html=True)
                elif event["type"] == "final_text":
                    final_response = event["content"]
                    response_placeholder.markdown(
                        f"<div class='final-response'><h4>✅ Result</h4><div>{final_response}</div></div>",
                        unsafe_allow_html=True,
                    )

        # Fallback if no final text
        if not final_response:
            final_response = (
                "I processed the request using my tools, "
                "but I don't have a final text response to share yet."
            )
            response_placeholder.markdown(
                f"<div class='final-response'><h4>ℹ️ Note</h4><div>{final_response}</div></div>",
                unsafe_allow_html=True,
            )

    # Prepare history entry (preserve tool activity via HTML details)
    full_response_for_history = ""
    if tool_activity_md:
        history_expander_content = "\n\n---\n\n".join(tool_activity_md)
        full_response_for_history += f"""
<details>
  <summary>🔎 Show Agent Activity</summary>

{history_expander_content}

</details>
"""
    full_response_for_history += final_response

    # Store assistant message
    st.session_state.messages.append({"role": "assistant", "content": full_response_for_history})
