"""
AeroGuard – LangGraph Multi-Agent AI Engine (MCP Client)
=========================================================
Upgraded to act as an MCP Client.  Instead of receiving a massive
pre-filled CityState, agents now use MCP tool-calling to fetch
exactly the data they need in real time.

Agent 1 (Anomaly Detector)  → calls `fetch_openaq_live` via MCP
Agent 2 (Source Attributor) → calls `get_wind_patterns` via MCP,
                              reads `municipal_grievances` resource
Agent 3 (Enforcement Agent) → no MCP (acts on state from Agents 1 & 2)
"""

import json
import os
import sys
import asyncio
from typing import TypedDict
from pathlib import Path

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools


# ---------------------------------------------------------------------------
# State schema — now slim: only trigger context + agent-populated fields
# ---------------------------------------------------------------------------

class CityState(TypedDict):
    """State schema for the Urban Air Quality multi-agent workflow."""
    # --- Trigger context (provided by FastAPI) ---
    target_location: str            # e.g., "East, Delhi"
    alert_context: str              # e.g., "AQI Spike Detected – PM2.5 at 347"
    timestamp: str                  # ISO timestamp of the trigger
    # --- Populated by agents via MCP tool calls ---
    sensor_data: list[dict]         # Filled by Agent 1
    wind_direction: str             # Filled by Agent 2
    wind_speed_kmh: float           # Filled by Agent 2
    regional_grievances: list[dict] # Filled by Agent 2
    is_hazardous: bool              # Set by Agent 1
    anomaly_summary: str            # Set by Agent 1
    identified_source: str          # Set by Agent 2
    enforcement_order: dict         # Set by Agent 3


# ---------------------------------------------------------------------------
# MCP connection helper
# ---------------------------------------------------------------------------
MCP_SERVER_SCRIPT = str(Path(__file__).resolve().parent / "mcp_server.py")


async def _run_with_mcp_tools(callback):
    """Connect to the MCP server via stdio, load tools, and run callback."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER_SCRIPT],
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
        },
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            return await callback(tools, session)


def _run_mcp_sync(callback):
    """Synchronous wrapper for async MCP operations."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an existing event loop (e.g., FastAPI)
        # Use a new thread to run the async code
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, _run_with_mcp_tools(callback))
            return future.result()
    else:
        return asyncio.run(_run_with_mcp_tools(callback))


# ---------------------------------------------------------------------------
# Agent 1: Anomaly Detector (MCP-enabled)
# ---------------------------------------------------------------------------

def anomaly_detector_node(state: CityState) -> dict:
    """Analyze live air quality by calling fetch_openaq_live via MCP,
    then determine if conditions are hazardous."""

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    # ---- Step 1: Fetch live data via MCP ----
    city = state["target_location"].split(",")[0].strip()

    async def fetch_data(tools, session):
        # Find the fetch_openaq_live tool
        fetch_tool = None
        for t in tools:
            if t.name == "fetch_openaq_live":
                fetch_tool = t
                break

        if fetch_tool is None:
            return {"readings": [], "error": "fetch_openaq_live tool not found"}

        result = await fetch_tool.ainvoke({"city": city, "parameter": "pm25"})
        return json.loads(result) if isinstance(result, str) else result

    try:
        openaq_data = _run_mcp_sync(fetch_data)
    except Exception as e:
        openaq_data = {"readings": [], "error": str(e)}

    readings = openaq_data.get("readings", [])

    # Build sensor_data from MCP results
    sensor_data = []
    for r in readings:
        sensor_data.append({
            "station": r.get("station", "Unknown"),
            "AQI": r.get("AQI_approx", 0),
            "PM2_5": r.get("PM2_5", 0),
            "PM10": r.get("PM10_approx", 0),
        })

    if not sensor_data:
        return {
            "sensor_data": [],
            "is_hazardous": False,
            "anomaly_summary": f"No live sensor data available for {city}. {openaq_data.get('error', '')}",
        }

    # ---- Step 2: LLM analysis of the fetched data ----
    prompt = PromptTemplate(
        input_variables=["sensor_data"],
        template=(
            "You are an environmental air-quality monitor.\n"
            "Analyze the following regional sensor data and determine whether "
            "any PM2.5 or AQI values exceed safe limits (AQI > 200 is hazardous).\n\n"
            "Sensor data:\n{sensor_data}\n\n"
            "Return ONLY valid JSON with exactly two keys:\n"
            '  "is_hazardous": true or false,\n'
            '  "summary": "<one-sentence explanation>"\n'
            "Do not include any text outside the JSON object."
        ),
    )

    chain = prompt | llm
    response = chain.invoke({"sensor_data": json.dumps(sensor_data, indent=2)})

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {"is_hazardous": False, "summary": "Unable to parse sensor analysis."}

    return {
        "sensor_data": sensor_data,
        "is_hazardous": result.get("is_hazardous", False),
        "anomaly_summary": result.get("summary", ""),
    }


# ---------------------------------------------------------------------------
# Agent 2: Source Attributor (MCP-enabled)
# ---------------------------------------------------------------------------

def source_attributor_node(state: CityState) -> dict:
    """Use atmospheric dispersion reasoning. Calls get_wind_patterns via MCP
    and reads municipal_grievances resource to correlate sources."""

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    # ---- Step 1: Fetch wind patterns + grievances via MCP ----
    district = state["target_location"].split(",")[0].strip()

    async def fetch_context(tools, session):
        wind_data = {}
        grievances = []

        # Call get_wind_patterns tool
        for t in tools:
            if t.name == "get_wind_patterns":
                result = await t.ainvoke({"latitude": 28.6139, "longitude": 77.2090})
                wind_data = json.loads(result) if isinstance(result, str) else result
                break

        # Read grievances resource
        try:
            resource_uri = f"municipal_grievances://{district}"
            resource_content = await session.read_resource(resource_uri)
            # resource_content is a ReadResourceResult with contents list
            for content in resource_content.contents:
                text = content.text if hasattr(content, "text") else str(content)
                grievance_data = json.loads(text)
                grievances = grievance_data.get("grievances", [])
        except Exception:
            # Fallback: try "all"
            try:
                resource_content = await session.read_resource("municipal_grievances://all")
                for content in resource_content.contents:
                    text = content.text if hasattr(content, "text") else str(content)
                    grievance_data = json.loads(text)
                    grievances = grievance_data.get("grievances", [])[:5]
            except Exception:
                grievances = []

        return wind_data, grievances

    try:
        wind_data, grievances = _run_mcp_sync(fetch_context)
    except Exception as e:
        wind_data = {
            "wind_speed_kmh": 0,
            "wind_direction_compass": "N/A",
            "error": str(e),
        }
        grievances = []

    wind_direction = wind_data.get("wind_direction_compass", "N/A")
    wind_speed = wind_data.get("wind_speed_kmh", 0)

    # ---- Step 2: LLM source attribution ----
    prompt = PromptTemplate(
        input_variables=[
            "anomaly_summary",
            "wind_direction",
            "wind_speed_kmh",
            "regional_grievances",
        ],
        template=(
            "You are an atmospheric-dispersion analyst for urban air quality.\n\n"
            "## Atmospheric Dispersion Reasoning\n"
            "Pollutants travel DOWNWIND from their source. Therefore, when a "
            "pollution spike is detected at a monitoring station, the true source "
            "lies UPWIND — i.e., in the opposite direction of the prevailing wind. "
            "Higher wind speeds carry pollutants farther from their origin, while "
            "lower speeds indicate a nearby source.\n\n"
            "Follow these steps:\n"
            "1. DETERMINE THE UPSTREAM DIRECTION: The wind is blowing FROM "
            "{wind_direction} at {wind_speed_kmh} km/h. Compute the opposite "
            "compass direction — that is where the source must be located.\n"
            "2. SCAN REGIONAL GRIEVANCES: Review the citizen complaints below and "
            "identify any events, industrial activity, construction, or burning "
            "reported in or near the upstream direction.\n"
            "3. DEDUCE THE SOURCE: Correlate the anomaly details with the upstream "
            "grievances to name the most likely source of the pollution spike.\n\n"
            "--- Data Inputs ---\n"
            "Anomaly summary: {anomaly_summary}\n\n"
            "Wind: {wind_direction} at {wind_speed_kmh} km/h\n\n"
            "Regional grievances:\n{regional_grievances}\n\n"
            "--- Instructions ---\n"
            "Return ONLY a 2-sentence explanation:\n"
            "  Sentence 1: State the upstream direction and the matching grievance.\n"
            "  Sentence 2: Name the deduced pollution source and briefly justify it.\n"
            "Do not include any other text."
        ),
    )

    chain = prompt | llm
    response = chain.invoke({
        "anomaly_summary": state["anomaly_summary"],
        "wind_direction": wind_direction,
        "wind_speed_kmh": wind_speed,
        "regional_grievances": json.dumps(grievances, indent=2),
    })

    return {
        "wind_direction": wind_direction,
        "wind_speed_kmh": wind_speed,
        "regional_grievances": grievances,
        "identified_source": response.content.strip(),
    }


# ---------------------------------------------------------------------------
# Agent 3: Enforcement Agent (no MCP — uses state from Agents 1 & 2)
# ---------------------------------------------------------------------------

def enforcement_agent_node(state: CityState) -> dict:
    """Generate a municipal enforcement dispatch order based on the identified source."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    prompt = PromptTemplate(
        input_variables=["identified_source", "target_location"],
        template=(
            "You are the City Commissioner responsible for environmental enforcement "
            "in {target_location}.\n\n"
            "A pollution source has been identified:\n"
            "{identified_source}\n\n"
            "You MUST follow these three strict dispatch rules:\n"
            "1. If the source is WASTE BURNING → dispatch a Fire Inspector to the "
            "site immediately and issue a cease-and-desist notice.\n"
            "2. If the source is CONSTRUCTION DUST → mandate water sprinkling at the "
            "construction site and assign the Building Compliance Department.\n"
            "3. If the source is VEHICULAR TRAFFIC → reroute heavy trucks away from "
            "the affected corridor and assign the Traffic Police Department.\n\n"
            "If the source does not match any of the above, use your best judgement "
            "to assign the most appropriate department and action.\n\n"
            "Return ONLY a valid JSON object — the Dispatch Order — with exactly "
            "these four keys:\n"
            '  "Action_Type": "<the specific enforcement action to take>",\n'
            '  "Assigned_Department": "<department or authority responsible>",\n'
            '  "Priority_Level": "Critical" | "High" | "Medium" | "Low",\n'
            '  "Justification": "<1-sentence reason for this dispatch>"\n\n'
            "Do not include any text outside the JSON object."
        ),
    )

    chain = prompt | llm
    response = chain.invoke({
        "identified_source": state["identified_source"],
        "target_location": state["target_location"],
    })

    # Parse the LLM's JSON response
    try:
        order = json.loads(response.content)
    except json.JSONDecodeError:
        order = {
            "Action_Type": "Manual review required",
            "Assigned_Department": "Environmental Office",
            "Priority_Level": "High",
            "Justification": "Automated dispatch parsing failed; escalating for manual review.",
        }

    return {
        "enforcement_order": order,
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def route_after_anomaly(state: CityState) -> str:
    """Conditional edge: continue only if hazardous conditions are detected."""
    return "source_attributor" if state["is_hazardous"] else END


workflow = StateGraph(CityState)

# Register nodes
workflow.add_node("anomaly_detector", anomaly_detector_node)
workflow.add_node("source_attributor", source_attributor_node)
workflow.add_node("enforcement_agent", enforcement_agent_node)

# Entry point
workflow.set_entry_point("anomaly_detector")

# Conditional edge: anomaly_detector → source_attributor OR END
workflow.add_conditional_edges(
    "anomaly_detector",
    route_after_anomaly,
    {
        "source_attributor": "source_attributor",
        END: END,
    },
)

# Linear edges for the rest of the pipeline
workflow.add_edge("source_attributor", "enforcement_agent")
workflow.add_edge("enforcement_agent", END)

# Compile the graph
aeroguard_app = workflow.compile()


# ---------------------------------------------------------------------------
# Local test harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

    mock_input: CityState = {
        "target_location": "East, Delhi",
        "alert_context": "AQI Spike Detected – PM2.5 readings above 200",
        "timestamp": "2025-01-15T08:30:00",
        # --- These will be populated by agents via MCP ---
        "sensor_data": [],
        "wind_direction": "",
        "wind_speed_kmh": 0.0,
        "regional_grievances": [],
        "is_hazardous": False,
        "anomaly_summary": "",
        "identified_source": "",
        "enforcement_order": {},
    }

    print("=" * 60)
    print("  AeroGuard – MCP-Enabled Multi-Agent Workflow")
    print("=" * 60)

    result = aeroguard_app.invoke(mock_input)

    print(f"\n📍 Location         : {result['target_location']}")
    print(f"⚠️  Hazardous        : {result['is_hazardous']}")
    print(f"📋 Anomaly Summary  : {result['anomaly_summary']}")
    print(f"📡 Sensor Stations  : {len(result.get('sensor_data', []))}")
    print(f"🌬️  Wind             : {result.get('wind_direction', 'N/A')} at {result.get('wind_speed_kmh', 0)} km/h")
    print(f"📢 Grievances Used  : {len(result.get('regional_grievances', []))}")
    print(f"🔍 Identified Source: {result.get('identified_source', 'N/A')}")
    print(f"🚨 Enforcement Order: {json.dumps(result.get('enforcement_order', {}), indent=2)}")