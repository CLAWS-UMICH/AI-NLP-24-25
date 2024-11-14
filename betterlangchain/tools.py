# tools.py
from langchain.agents import Tool

def get_hardcoded_vitals(input_text=None):
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

def get_system_status(input_text=None):  # Accept an argument to match the expected signature
    status = {
        "cpu_usage": "45%",
        "memory_usage": "60%",
        "disk_space": "120GB free out of 256GB",
        "uptime": "5 days, 4 hours, 23 minutes"
    }
    return status

system_status_tool = Tool(
    name="GetSystemStatus",
    func=get_system_status,
    description="Returns a mock system status including CPU usage, memory usage, disk space, and uptime."
)

tools = [vitals_tool, system_status_tool]
