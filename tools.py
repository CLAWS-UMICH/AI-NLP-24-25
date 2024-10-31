# tools.py

def get_hardcoded_vitals():
    """
    Function that returns a hardcoded set of vitals.
    This can be used as a tool in LangChain for testing purposes.
    """
    vitals = {
        "heart_rate": "72 bpm",
        "blood_pressure": "120/80 mmHg",
        "temperature": "98.6°F",
        "respiration_rate": "16 breaths per minute"
    }
    return vitals
