# tools.py
import random
from dataclasses import dataclass

@dataclass
class Tool:
    name: str
    description: str 
    params: list
    return_description: str
    function: callable

def get_vitals(input_text=None):
    print("Getting hardcoded vitals... (5 second delay)")
    vitals = {
        "heart_rate": f"{random.randint(60,100)} bpm",
        "blood_pressure": f"{random.randint(90,140)}/{random.randint(60,90)} mmHg",
        "temperature": f"{round(random.uniform(97.0,99.5),1)}°F",
        "respiration_rate": f"{random.randint(12,20)} breaths per minute"
    }
    print("Vitals retrieved!")
    return vitals

def get_system_status(input_text=None):
    print("Getting system status... (5 second delay)")
    status = {
        "cpu_usage": f"{random.randint(20,90)}%",
        "memory_usage": f"{random.randint(30,95)}%",
        "disk_space": f"{random.randint(50,200)}GB free out of 256GB", 
        "uptime": f"{random.randint(1,30)} days, {random.randint(0,23)} hours, {random.randint(0,59)} minutes"
    }
    print("Status retrieved!")
    return status

tools = [
    Tool(
        name="GetVitals",
        description="Returns a set of vitals: heart rate, blood pressure, temperature, and respiration rate.",
        params=[],
        return_description="Dictionary containing vital signs",
        function=get_vitals
    ),
    Tool(
        name="GetSystemStatus", 
        description="Returns the current system status including CPU usage, memory usage, disk space, and uptime.",
        params=[],
        return_description="Dictionary containing system status information",
        function=get_system_status
    )
]
