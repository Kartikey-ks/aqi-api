import json
import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate


class CityState(TypedDict):
    """State schema for the Urban Air Quality multi-agent workflow."""
    target_location: str
    sensor_data: list[dict]           # Regional AQI data from sensors
    wind_direction: str
    wind_speed_kmh: float
    regional_grievances: list[dict]   # Citizen complaints
    is_hazardous: bool
    anomaly_summary: str
    identified_source: str
    enforcement_order: dict


def anomaly_detector_node(state: CityState) -> dict:
    """Analyze sensor data for hazardous AQI/PM2.5 levels using an LLM."""
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0,
        api_key=os.environ.get("GROQ_API_KEY"),
    )

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
    response = chain.invoke({"sensor_data": json.dumps(state["sensor_data"], indent=2)})

    # Parse the LLM's JSON response
    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback: treat as non-hazardous if parsing fails
        result = {"is_hazardous": False, "summary": "Unable to parse sensor analysis."}

    return {
        "is_hazardous": result.get("is_hazardous", False),
        "anomaly_summary": result.get("summary", ""),
    }


def source_attributor_node(state: CityState) -> dict:
    """Use atmospheric dispersion reasoning to identify the pollution source."""
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0,
        api_key=os.environ.get("GROQ_API_KEY"),
    )

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
        "wind_direction": state["wind_direction"],
        "wind_speed_kmh": state["wind_speed_kmh"],
        "regional_grievances": json.dumps(state["regional_grievances"], indent=2),
    })

    return {
        "identified_source": response.content.strip(),
    }


def enforcement_agent_node(state: CityState) -> dict:
    """Generate a municipal enforcement dispatch order based on the identified source."""
    llm = ChatGroq(
        model="llama3-70b-8192",
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
    mock_input: CityState = {
        "target_location": "Anand Vihar, Delhi",
        "sensor_data": [
            {"station": "Anand Vihar", "AQI": 347, "PM2_5": 198.4, "PM10": 312.0},
            {"station": "ITO", "AQI": 185, "PM2_5": 102.1, "PM10": 201.5},
            {"station": "Dwarka Sec-8", "AQI": 120, "PM2_5": 55.3, "PM10": 98.7},
        ],
        "wind_direction": "NW",
        "wind_speed_kmh": 12.5,
        "regional_grievances": [
            {
                "complaint_id": "GRV-1042",
                "location": "Ghazipur Landfill (SE of Anand Vihar)",
                "type": "Waste Burning",
                "description": "Thick smoke observed from open waste burning at Ghazipur landfill.",
            },
            {
                "complaint_id": "GRV-1055",
                "location": "NH-24 Flyover Construction (East)",
                "type": "Construction Dust",
                "description": "Heavy dust clouds from ongoing flyover construction on NH-24.",
            },
        ],
        "is_hazardous": False,
        "anomaly_summary": "",
        "identified_source": "",
        "enforcement_order": {},
    }

    print("=" * 60)
    print("  AeroGuard – Urban Air Quality Multi-Agent Workflow")
    print("=" * 60)

    result = aeroguard_app.invoke(mock_input)

    print(f"\n📍 Location        : {result['target_location']}")
    print(f"⚠️  Hazardous       : {result['is_hazardous']}")
    print(f"📋 Anomaly Summary : {result['anomaly_summary']}")
    print(f"🔍 Identified Source: {result.get('identified_source', 'N/A')}")
    print(f"🚨 Enforcement Order: {json.dumps(result.get('enforcement_order', {}), indent=2)}")
