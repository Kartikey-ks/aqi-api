# AeroGuard: AI-Powered Urban Air Quality Intelligence Platform

AeroGuard is an enterprise-grade, Agentic AI platform designed to help city administrators transition from reactive air quality monitoring to proactive, evidence-based interventions. 

By fusing continuous monitoring data, meteorological forecasts, and geospatial layers, this platform gives municipal bodies the intelligence to reduce pollution at the source rather than just measure it.

---

## 🚀 Solution Approach

Current municipal systems suffer from a lack of actionable intelligence. While IoT sensors gather data, there is no automated layer to attribute pollution sources or deploy inspectors. AeroGuard solves this using a **LangGraph Multi-Agent System** integrated with a **Model Context Protocol (MCP)** server.

1. **Lean Trigger Layer:** A FastAPI backend acts as the gateway, constantly monitoring standard AQI thresholds. If the AQI exceeds safe limits, it triggers the multi-agent workflow.
2. **Context Optimization & Spatial Filtering:** Before massive payloads from OpenAQ and Open-Meteo reach the LLM, the backend performs strict spatial filtering. Data is geofenced to the exact queried city/ward. This highly optimized context window ensures the AI receives only high-signal data, reducing latency, saving tokens, and preventing geographic hallucinations.
3. **Agentic Orchestration:** * **Agent 1 (Anomaly Detector):** Analyzes filtered environmental data to validate localized spikes and hazardous thresholds.
    * **Agent 2 (Source Attributor):** Uses MCP tools to pull wind patterns and citizen grievance data, correlating upstream direction to pinpoint the likely pollution source (e.g., illegal construction, waste burning).
    * **Agent 3 (Enforcement Agent):** Formulates a structured JSON action plan based on standard municipal policy playbooks.
4. **GenAI Generation:** A Groq-powered LLM translates the structured multi-agent output into human-readable insights, generating Geospatial Dispatch Orders, Public Advisories, and a transparent AI Reasoning Audit Trail.
5. **Interactive UI:** The final insights are served to a React/Gradio frontend, plotting exact intervention zones on an interactive city map.

---

## 🛠️ Tech Stack

### AI & Machine Learning
* **AI Orchestration:** LangGraph (State machines and multi-agent routing)
* **LLM Engine:** Groq (High-speed inference for GenAI outputs)
* **Predictive Modeling:** XGBoost (For predictive AQI modeling and 3-day trend forecasting)

### Backend & Tooling
* **Backend Framework:** FastAPI (Python)
* **Tool Integration:** Model Context Protocol (MCP) Server for fetching external APIs 
* **Data Sources:** OpenAQ (Live AQI), Open-Meteo (Wind/Weather), CPCB Historical Data, Mock Citizen Grievances (JSON)

### Frontend Compatibility
* Designed to interface seamlessly with **React** and **Gradio** web applications via REST APIs.

---

## ⚙️ Setup & Installation

**1. Clone the repository**
Bash
git clone [https://github.com/Kartikey-ks/aqi-api.git](https://github.com/Kartikey-ks/aqi-api.git)
cd aqi-api

2. Create and activate a virtual environment
Bash
# macOS/Linux
python -m venv venv
source venv/bin/activate  
# Windows
venv\Scripts\activate

3. Install dependencies
Bash
pip install -r requirements.txt

4. Environment Variables
Create a .env file in the root directory and configure your essential API keys:
Code snippet
GROQ_API_KEY=your_groq_api_key_here
OPENAQ_API_KEY=your_openaq_api_key_here
WEATHER_API_KEY=your_weather_api_key_here


💻 Usage

1. Start the FastAPI Server
Run the backend server using Uvicorn.
Bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000

2. API Endpoints
The primary entry point for the frontend dashboard is the analyze endpoint.
Endpoint: GET /analyze-city
Parameters: city (string) - e.g., ?city=Delhi
Response Handling:
Safe AQI: Returns real-time live stats (Green Map state).
Hazardous AQI: Triggers the LangGraph engine and returns a comprehensive JSON payload containing the Dispatch Order, Audit Trail, and Public Advisory drafted by the LLM.
Example Request:
Bash
curl -X GET "http://localhost:8000/analyze-city?city=Delhi"


Architecture Note
The system utilizes an Environmental MCP Server functioning as a tool provider. The LangGraph engine acts as the MCP client, autonomously calling tools like fetch_openaq_live(), get_wind_patterns(), and fetch_grievances().

This separation of concerns ensures that the AI logic is completely modular. New civic data streams (like satellite imagery or traffic feeds) can be added to the MCP server in the future without altering the core agentic orchestration.
