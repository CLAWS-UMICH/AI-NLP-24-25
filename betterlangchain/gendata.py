from better import Agent, Tool, Param
from tools import get_vitals, get_system_status
import os
import json
import time

conversation_folder = "screen_conversation_logs"
output_file = "my_output_destination.json"

tools = [
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
    ]

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
]

def main():
    # Create agent with tools (currently commented out for data generation)
    agent = Agent(tools, conversation_folder=conversation_folder)
    print(f"Processing {len(test_prompts)} diverse prompts...")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nProcessing prompt {i}/{len(test_prompts)}: {prompt}")
        agent.clear_conversation_history()
        # Note that the agent saves the conversation history to the conversation_folder itself
        response = agent.ask(prompt)
        print("Response received and saved ------------------------------------------------------------------------")

    # Read all the files in the conversation_folder and format them together into training samples
    conversation_files = [f for f in os.listdir(conversation_folder) if f.endswith('.json')]
    
    formatted_conversations = []
    
    for conv_file in conversation_files:
        with open(os.path.join(conversation_folder, conv_file)) as f:
            conv_data = json.load(f)
        
        # We'll build a running context. For every assistant message (that isn't a tool call) seen,
        # we create a sample where the instruction is the conversation history (all previous messages
        # including any tool call formatting) and the output is the current assistant reply.
        context = []
        for msg in conv_data:
            if msg.get("from_") == "user":
                context.append(f"user: {msg['content']}")
            elif msg.get("tool_used"):
                # Format the tool call in two parts (function call and its result)
                context.append(f"assistant: Function Call: {msg['tool_used']}({json.dumps(msg.get('params', {}))})")
                context.append(f"Result: {json.dumps(msg.get('return_value', {}))}")
            elif msg.get("from_") == "ai":
                # When we hit an assistant message (that is not a tool call) we create a sample.
                if msg.get("content"):
                    sample = {
                        "instruction": "\n".join(context),
                        "input": "",
                        "output": msg["content"]
                    }
                    formatted_conversations.append(sample)
                # Then add the assistant message to the running context.
                context.append(f"assistant: {msg.get('content', '')}")
    
    # Save all training samples to a single JSON file
    with open(os.path.join(conversation_folder, output_file), "w") as f:
        json.dump(formatted_conversations, f, indent=2)

if __name__ == "__main__":
    main()
