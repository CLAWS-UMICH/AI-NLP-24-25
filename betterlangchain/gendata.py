from better import Agent, Tool, Param
from tools import get_vitals, get_system_status

# List of diverse prompts for testing agent responses
test_prompts = [
    # Basic vitals queries
    "What are my vitals?",
    "What is my heart rate?",
    "What is my blood pressure?",
    "What is my temperature?",
    "What is my respiration rate?",
    
    # System status queries 
    "Check my CPU usage",
    "What is the system uptime?",
    "How much memory is being used?",
    "What's the disk space status?",
    "Give me the full system status",
    
    # Combined queries
    "Can you give me my vitals and system status?",
    "Show me both health and system metrics",
    "What are all my stats?",
    "Give me a complete status report",
    "Check everything and report back",
    
    # Specific combinations
    "Tell me my heart rate and CPU usage",
    "What's my blood pressure and memory usage?",
    "Check my temperature and disk space",
    "Report on respiration and uptime",
    
    # Direct questions
    "Is the CPU running hot?",
    "Am I running a fever?",
    "Is my heart rate elevated?",
    "How's the memory holding up?",
    
    # Status requests
    "Run a system check",
    "Do a health check",
    "Full diagnostic please",
    "Status update needed",
    "Give me the latest readings",
    
    # Focused queries
    "Just the vitals please",
    "System metrics only",
    "Health stats report",
    "Computer status only",
    
    # Multiple metrics
    "Heart rate and blood pressure check",
    "CPU and memory status",
    "Temperature and respiration check",
    "Disk space and uptime report",
    
    # General inquiries
    "How am I doing?",
    "How's the system running?",
    "Everything okay?",
    "Status report"
# ]

# old_prompts = [
    # Basic health and system queries
    "How am I doing?",
    "Run a full diagnostic",
    "Status report",
    "What's my current state?",
    
    # Natural health-focused queries
    "I'm feeling a bit dizzy, can you check my vitals?",
    "My heart feels like it's racing, what's my pulse?",
    "Do my vitals look normal?",
    "I just finished exercising, how's my heart rate?",
    "Been feeling warm, is my temperature okay?",
    "Can you check my blood pressure?",
    "How's my breathing rate looking?",
    
    # System-focused queries
    "The computer feels sluggish",
    "How much disk space is left?",
    "Is the system running efficiently?",
    "The fans are loud, is CPU usage high?",
    "Check if we're low on memory",
    "How's the system performing?",
    "What's our current memory usage?",
    
    # Combined complex queries
    "I'm feeling tired and the system's acting weird - what's going on?",
    "Full health and system analysis please",
    "Something's not right - check everything",
    "The air feels stuffy and the computer's slow, run all diagnostics",
    "Check both my vitals and the system status",
    
    # Urgent requests
    "Need a status check immediately",
    "System and vital check right now",
    "Urgent check on everything",
    "Quick, check all my vitals and the system",
    
    # Analytical/Detailed requests
    "Give me a detailed analysis of my vitals and system metrics",
    "I need a full report on my health and system status",
    "Tell me everything about my current state and system performance",
    "What's the complete status of both me and the computer?",
    
    # Casual queries
    "Hey, how's everything looking?",
    "Can you do a quick check of everything?",
    "What's the status on all fronts?",
    "Give me an update on everything",
    
    # Specific combinations
    "The system's running hot and I'm not feeling well",
    "Check both CPU usage and my vital signs",
    "How's the computer running and how are my vitals?",
    "System performance and health check please",
    
    # Concerned queries
    "I'm worried about the system and my health",
    "Something feels off with both me and the computer",
    "Not feeling right and the system's acting strange",
    "Check if everything's okay with me and the computer",
]

def main():
    # Create agent with tools
    agent = Agent([
        Tool(
            name="GetHardcodedVitals",
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
    ])

    print(f"Processing {len(test_prompts)} diverse prompts...")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nProcessing prompt {i}/{len(test_prompts)}: {prompt}")
        agent.clear_conversation_history()
        response = agent.ask(prompt)
        print(f"Response received and saved ------------------------------------------------------------------------")

if __name__ == "__main__":
    main()
