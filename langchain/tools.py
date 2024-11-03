# tools.py
from langchain.agents import Tool

def get_hardcoded_vitals():
    vitals = {
        "heart_rate": "72 bpm",
        "blood_pressure": "120/80 mmHg",
        "temperature": "98.6°F",
        "respiration_rate": "16 breaths per minute"
    }
    return vitals

vitals_tool = Tool(
    name="GetHardcodedVitals",
    func=get_hardcoded_vitals,
    description="Returns a hardcoded set of vitals: heart rate, blood pressure, temperature, and respiration rate."
)

tools = [vitals_tool]
