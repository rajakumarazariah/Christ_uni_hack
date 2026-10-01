# Crisis Command 🚨

A fast emergency response coordination prototype for municipal dispatch centers.

- **Deterministic Hungarian Allocation**: Unit-to-incident assignment uses SciPy (`linear_sum_assignment`) minimizing expected harm with configurable switch penalties.
- **Strict Role Separation**: The LLM (Gemini 2.5 Flash) **never** decides assignments. It is restricted to extracting structured incident facts from text/voice dispatches and phrasing explanations.
- **Human-in-the-Loop Safeguards**: LangGraph pauses execution when high-risk operations occur (diverting en-route units, critical severity-5 deficits, or unconfirmed triage data).
- **Control Room Console**: Vanilla JS + Leaflet (no build step, no npm), rendering real-time unit positions, response routes, ghost tracks of previous plans, and spatial coverage gaps.

---

## 1. Quickstart with `uv`

### Prerequisites
- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) installed:
  ```bash
  curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
Installation
Bash
# Clone the repository
git clone [https://github.com/your-org/crisis-command.git](https://github.com/your-org/crisis-command.git)
cd crisis-command

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Configure environment
cp .env.example .env
Edit .env and add your GEMINI_API_KEY:

Ini, TOML
GEMINI_API_KEY=your_actual_key_here
LLM_MOCK=false
(If GEMINI_API_KEY is omitted, the app starts with LLM_MOCK=true automatically).

Run the Server
Bash
uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
Open http://127.0.0.1:8000 in your browser.

Run Tests
Bash
uv run pytest -v
2. 6-Step Hackathon Demo Script
Inspect Opening Console & State:

Navigate to http://127.0.0.1:8000.

Observe 11 units deployed across Tiruchirappalli, 4 hospitals, 3 shelters, and the green/amber coverage deficit heatmap.

The initial plan routes ambulances and rescue teams to the 3 seed incidents.

Examine the Opening Plan & Ghost Tracks:

Toggle Ghost Plan: ON/OFF in the top bar.

The initial dispatch lines show optimal assignments computed by the Hungarian solver.

Multimodal Intake (Tamil / Multilingual Text):

In the left panel, paste:

"சத்திரம் பேருந்து நிலையம் அருகில் 3 பேர் காயமடைந்த விபத்து"
(Or: "Severe accident near Chattiram Bus Stand with 3 injured")

Click Process Report.

Gemini parses the language, extracts major_trauma, severity 4, and fuzzy-matches Chattiram Bus Stand via landmarks.json. The solver replans immediately.

Voice Dispatch Ingestion:

Click the Microphone icon. Speak an incident:

"Emergency near Rockfort Temple, heavy smoke coming from a commercial building."

Click to stop. The audio stream uploads to /api/incidents/audio, Gemini extracts the incident, and resolves coordinates.

Inject Critical Incident & Trigger Unit Breakdown:

Click Add Incident (Click) on the top bar and click anywhere near the center of the map. Select Cardiac Arrest, Severity 5.

Click on an active ambulance marker (e.g., AMB-1) and click Mark Unavailable (simulating engine breakdown).

Review Operator Diversion & Export SitRep:

The Hungarian solver diverts an en-route ambulance to the severity-5 cardiac scene.

Approval Modal Appears: Because an active en-route unit is diverted, LangGraph halts execution.

Review the reasoning card: ETA before vs. ETA after, harm reduction, and the abandoned incident penalty.

Click Approve Diversion Plan (or Reject to execute a non-diverting fallback solve).

Click Export SitRep on the top bar to download the complete METHANE Situation Report in Markdown.

3. Architecture & Separation of Concerns
[ Operator Voice / Text / Map Click ]
                 │
                 ▼
       backend/agents/intake.py
  (Gemini Extracts Facts -> No Assignment Powers)
                 │
                 ▼
       backend/logistics.py
  (OSRM / Haversine Matrix + Hazard Exclusion Zones)
                 │
                 ▼
         backend/solver.py
  (SciPy Hungarian Algorithm -> Deterministic Allocation)
                 │
                 ▼
       backend/agents/command.py
  (Computes Plan Diff + Numeric Cards + Risk Flags)
                 │
                 ▼
        backend/agents/graph.py
  (LangGraph Human Gate: Pauses on En-Route Diversions)
                 │
                 ▼
       backend/state.py (Store) ──SSE──► Leaflet UI Console
4. Troubleshooting
1. OSRM Service Unavailable or Timed Out
Symptom: Logs show OSRM connection errors or requests take > 3 seconds.

Resolution: Crisis Command defaults to great-circle Haversine distance with a 1.4x urban detour factor and AVG_SPEED_KMPH=40 fallback. To force offline routing entirely, set USE_OSRM=false in .env.

2. Gemini Quota Exceeded or No Internet
Symptom: 429 ResourceExhausted or timeout from Google GenAI client.

Resolution: Set LLM_MOCK=true in .env and restart. The system switches to regex/keyword rules (supporting English, Tamil, and Hindi medical terms) and deterministic template explanations.

3. Microphone Access Denied (getUserMedia)
Symptom: Clicking the mic produces a browser error: Microphone permission denied.

Resolution: Browsers restrict WebRTC audio APIs strictly to secure contexts (https://) or http://localhost / http://127.0.0.1. If accessing the console remotely across a LAN, use an SSH tunnel:

Bash
ssh -L 8000:127.0.0.1:8000 user@remote-machine