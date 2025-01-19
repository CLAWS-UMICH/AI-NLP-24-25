from dataclasses import dataclass
import json
from openai import OpenAI
import time
import os
from dotenv import load_dotenv
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import random
load_dotenv()

@dataclass
class Param:
    name: str
    type: str
    description: str
    required: bool

@dataclass 
class Tool:
    name: str
    description: str
    params: list
    return_description: str
    function: callable

@dataclass
class Message:
    from_: str  # user/ai/system
    content: str

@dataclass
class FunctionCall:
    tool_used: str
    params: dict
    return_value: any

def default_openai_chat(messages):
    print("OPEN AI CALL, SENDING MESSAGE",messages)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        messages=[{"role": msg["role"], "content": msg["content"]} for msg in messages],
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=500,
        stream=False
    )
    
    return response.choices[0].message.content
    
class Agent:
    def __init__(self, tools, chat_model=None):
        print("Initializing Agent...")
        self.tools = {tool.name: tool for tool in tools}
        print(f"Loaded {len(tools)} tools: {', '.join(self.tools.keys())}")
        
        # Use default OpenAI chat if no model provided
        self.chat_model = chat_model or default_openai_chat
        print("Chat model initialized")
        
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        print("OpenAI client initialized")
        
        # Add built-in give_response tool
        self.tools["give_response"] = Tool(
            name="give_response",
            description="Return a response to the user",
            params=[Param("response", "str", "Response text to return to user", True)],
            return_description="None",
            function=lambda response: None
        )
        print("Added built-in give_response tool")

        # Initialize conversation history with system prompt
        self.conversation_history = []

        system_prompt = """You are an AI assistant. Use tools to help users.
        
        YOU MUST RESPOND WITH THIS FORMAT:
        PLAN - Your plan description
        ACTIONS:
        tool_name1||{"param1": "value1"}
        tool_name2||{"param1": "value1"}
        CONTINUE/DONE

        CRITICAL RULES: 
        1. Each action must be on a new line
        2. Use '||' to separate tool name from parameters
        3. Parameters must be valid JSON objects
        4. Write '{}' if no parameters needed
        5. List actions that can run in parallel together
        6. For dependent actions, use multiple CONTINUE cycles
        7. ONLY use give_response tool as your FINAL action after collecting ALL needed data
        8. Format data clearly and professionally in your final response
        9. YOU MUST ALWAYS HAVE A PLAN SECTION, ACTIONS SECTION, AND CONTINUE/DONE SECTION
        10. Be precise. If the user asks for a specific metric, don't give them more then what they need.
        11. If the users query is more general, give them a reasonable answer.

        GIVE_RESPONSE FORMAT:
        - Always use give_response as the final action after collecting data
        - Format the response with clear sections and labels
        - Include all relevant data from previous tool calls
        - Use proper spacing and newlines (\\n) for readability
        - Example format:
          give_response||{"response": "Here are the results:\\n\\nSection 1:\\n- Data point 1: value\\n- Data point 2: value\\n\\nSection 2:\\n- Data point 3: value\\n- Data point 4: value"}

        Available tools:
        """ + "\n".join(f"- {name}: {tool.description}" for name, tool in self.tools.items())

        self.conversation_history.append(Message(from_="system", content=system_prompt))
        print("System prompt added to conversation history")
        
        self.executor = ThreadPoolExecutor(max_workers=10)
        print("Thread pool executor initialized")
    
    def get_tools_used(self):
        tools = [item.tool_used for item in self.conversation_history if isinstance(item, FunctionCall)]
        print(f"Tools used in conversation: {tools}")
        return tools
    
    def clear_conversation_history(self):
        print("Clearing conversation history...")
        self.conversation_history = [self.conversation_history[0]]
        print("Conversation history cleared, kept system prompt")

    def format_conversation_history(self):
        print("Formatting conversation history...")
        formatted_messages = []
        
        for item in self.conversation_history:
            if isinstance(item, Message):
                role = {
                    "user": "user",
                    "ai": "assistant",
                    "system": "system"
                }.get(item.from_, "user")
                formatted_messages.append({
                    "role": role,
                    "content": item.content
                })
            else:  
                formatted_messages.append({
                    "role": "system",
                    "content": f"Tool '{item.tool_used}' was called with parameters {item.params} and returned: {item.return_value}"
                })
        
        print(f"Formatted {len(formatted_messages)} messages")
        return formatted_messages

    def execute_tool(self, tool_name, params):
        print(f"\nExecuting tool: {tool_name}")
        print(f"Parameters: {params}")
        
        if tool_name not in self.tools:
            print(f"ERROR: Unknown tool {tool_name}")
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool = self.tools[tool_name]
    
        
        result = tool.function(**params)
        print(f"Tool returned: {result}")
        
        self.conversation_history.append(FunctionCall(
            tool_used=tool_name,
            params=params,
            return_value=result
        ))
        print("Tool execution recorded in conversation history")
        
        return result

    def process_response(self):
        try:
            response = self.chat_model(
                messages=self.format_conversation_history(),
            )

            self.conversation_history.append(Message(from_="ai", content=response))

            print("OPEN AI CALL, LATEST RESPONSE",response)
            # Collect all actions
            actions = []
            current_section = None
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if line == "ACTIONS:":
                    current_section = "actions"
                    continue
                elif line in ["CONTINUE", "DONE"]:
                    break
                elif '||' in line and current_section == "actions":
                    try:
                        tool_name, params_str = line.split('||', 1)
                        tool_name = tool_name.strip()
                        params = json.loads(params_str)
                        actions.append((tool_name, params))
                    except Exception as e:
                        print(f"Error parsing action: {e}")
            
            # Execute all tools in parallel
            if actions:
                futures = []
                for tool_name, params in actions:
                    if tool_name == "give_response":
                        # Store give_response for last
                        final_response = params.get("response")
                    else:
                        # Execute other tools in parallel
                        future = self.executor.submit(self.execute_tool, tool_name, params)
                        futures.append(future)
                
                # Wait for all tools to complete if there are any
                if futures:
                    concurrent.futures.wait(futures)
                
                # If we had a give_response, return it after all tools complete
                if 'final_response' in locals():
                    save_conversation_history(self.conversation_history)
                    return final_response
            
            # No give_response found, continue the conversation
            return None
                
        except Exception as e:
            print(f"Error in response processing: {e}")
            raise

    def ask(self, user_input):
        print(f"\nReceived user input: {user_input}")
        self.conversation_history.append(Message(from_="user", content=user_input))
        
        while True:
            response = self.process_response()
            print("THIS IS THE LATEST RESPONSE",response)
            # If we got a response from give_response, return it
            if response is not None:
                return response
            
            # If we didn't get a response, continue the conversation
            continue
def save_conversation_history(conversation_history):
    print("Saving conversation history...")
    # Create conversation_logs directory if it doesn't exist
    os.makedirs('conversation_logs', exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join('conversation_logs', f'conversation_history_{timestamp}.json')
    with open(filepath, 'w') as f:
        json.dump([item.__dict__ for item in conversation_history], f, indent=2)
    print(f"Conversation history saved to {filepath}")


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

print("Creating agent with tools...")
agent = Agent([
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
    ])

def main(): 
    print("\nAI Assistant ready! Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            print("Exiting...")
            break
        print("\nProcessing request...")
        response = agent.ask(user_input)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()