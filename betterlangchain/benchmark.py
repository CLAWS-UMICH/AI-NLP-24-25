from old import langchain_agent_response, get_tools_used, clear_tools_used
from better import agent as better_agent
import time

# List of test prompts and expected tool calls
test_cases = [
    {
        "prompt": "What are my vital signs?",
        "expected_tools": ["GetHardcodedVitals"],
        "description": "Basic vitals check"
    },
    {
        "prompt": "How is the system performing?", 
        "expected_tools": ["GetSystemStatus"],
        "description": "Basic system status check"
    },
    {
        "prompt": "Tell me both my vitals and system status",
        "expected_tools": ["GetHardcodedVitals", "GetSystemStatus"],
        "description": "Combined vitals and system check"
    },
    {
        "prompt": "I'm feeling dizzy and the system is acting weird. What's going on?",
        "expected_tools": ["GetHardcodedVitals", "GetSystemStatus"],
        "description": "Natural language combined check"
    },
    {
        "prompt": "Run a complete health and system diagnostic",
        "expected_tools": ["GetHardcodedVitals", "GetSystemStatus"],
        "description": "Professional terminology check"
    },
    {
        "prompt": "My heart feels funny... check everything please!",
        "expected_tools": ["GetHardcodedVitals", "GetSystemStatus"],
        "description": "Urgent medical concern with implicit system check"
    },
    {
        "prompt": "Just give me all the data you have about my current state",
        "expected_tools": ["GetHardcodedVitals", "GetSystemStatus"],
        "description": "Vague but comprehensive request"
    },
    {
        "prompt": "vitals + sys status ASAP",
        "expected_tools": ["GetHardcodedVitals", "GetSystemStatus"],
        "description": "Terse command style input"
    },
    {
        "prompt": "I need a full report on both my health metrics and system performance, with particular attention to any anomalies",
        "expected_tools": ["GetHardcodedVitals", "GetSystemStatus"],
        "description": "Detailed analytical request"
    },
    {
        "prompt": "🫀❓🖥️❓",
        "expected_tools": ["GetHardcodedVitals", "GetSystemStatus"],
        "description": "Emoji-only query test"
    }
]

print("=== Running Benchmark ===")

# Benchmark better agent
better_total_accuracy = 0
better_total_time = 0

print("\nTesting Better Agent:")
for test in test_cases:
    # Clear history before each test
    better_agent.clear_conversation_history()
    
    start_time = time.time()
    better_response = better_agent.ask(test['prompt'])
    end_time = time.time()
    
    # Get unique tools used (excluding give_response)
    tools_used = set(tool for tool in better_agent.get_tools_used() 
                    if tool != "give_response")
    expected_tools = set(test['expected_tools'])
    
    # Calculate accuracy based on both precision and recall
    true_positives = len(tools_used & expected_tools)
    false_positives = len(tools_used - expected_tools)
    false_negatives = len(expected_tools - tools_used)
    
    if len(expected_tools) == 0:
        accuracy = 100.0 if len(tools_used) == 0 else 0.0
    else:
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / len(expected_tools)
        accuracy = (precision + recall) * 50  # Average of precision and recall, scaled to 100
    
    better_total_accuracy += accuracy
    better_total_time += (end_time - start_time)

# Benchmark old agent
old_total_accuracy = 0
old_total_time = 0

print("\nTesting Old Agent:")
for test in test_cases:
    start_time = time.time()
    print(test['prompt'])
    old_response = langchain_agent_response(test['prompt'])
    end_time = time.time()
    
    # Get tools used from old agent using new method
    tools_used = get_tools_used()
    expected_tools = test['expected_tools']
    
    # Calculate accuracy as percentage of correct tools used
    correct_tools = len(set(tools_used) & set(expected_tools))
    total_tools = len(expected_tools)
    accuracy = (correct_tools / total_tools) * 100
    
    old_total_accuracy += accuracy
    old_total_time += (end_time - start_time)
    
    # Clear tools used for next test
    clear_tools_used()

print("\n=== Results ===")
print("\nBetter Agent:")
print(f"Average accuracy: {better_total_accuracy / len(test_cases):.1f}%")
print(f"Average response time: {(better_total_time / len(test_cases)):.2f} seconds")

print("\nOld Agent:")
print(f"Average accuracy: {old_total_accuracy / len(test_cases):.1f}%")
print(f"Average response time: {(old_total_time / len(test_cases)):.2f} seconds")
