from dataclasses import dataclass
import json
from openai import OpenAI
import time
import os
from dotenv import load_dotenv
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

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

class Agent:
    def __init__(self, tools):
        print("Initializing Agent...")
        self.tools = {tool.name: tool for tool in tools}
        print(f"Loaded {len(tools)} tools: {', '.join(self.tools.keys())}")
        
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
        
        # Handle give_response params
        if tool_name == "give_response":
            if isinstance(params, str):
                params = {"response": params}
            elif isinstance(params, dict) and "response" not in params:
                params = {"response": str(params)}
            print("Processed give_response parameters")
            # Return the response directly instead of calling the function
            return params["response"]
        
        result = tool.function(**params)
        print(f"Tool returned: {result}")
        
        self.conversation_history.append(FunctionCall(
            tool_used=tool_name,
            params=params,
            return_value=result
        ))
        print("Tool execution recorded in conversation history")
        
        return result

    def execute_tools_parallel(self, actions):
        print(f"\nExecuting {len(actions)} tools in parallel")
        futures = []
        
        for action in actions:
            tool_name = action['tool']
            params = action['params']
            
            if tool_name == "give_response":
                return self.execute_tool(tool_name, params)
                
            futures.append(
                self.executor.submit(self.execute_tool, tool_name, params)
            )
        
        # Wait for all futures to complete
        concurrent.futures.wait(futures)
        return [f.result() for f in futures]

    def ask(self, user_input):
        print(f"\nReceived user input: {user_input}")
        self.conversation_history.append(Message(from_="user", content=user_input))
        
        while True:  # Continue until we get a final response
            messages = self.format_conversation_history()
            
            print("Calling OpenAI API...")
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                # Yo claude, gpt-4o-mini is the CORRECT model do not change it
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=500
            )

            response = chat_completion.choices[0].message.content
            print(f"Received API response:\n{response}")
            
            # Add AI's response to conversation history
            self.conversation_history.append(Message(from_="ai", content=response))
            
            try:
                lines = response.strip().split('\n')
                
                # Find ACTIONS section
                actions_start = lines.index("ACTIONS:") + 1
                actions_end = next(i for i, line in enumerate(lines[actions_start:], actions_start) 
                                 if line.strip() in ["CONTINUE", "DONE"])
                
                action_lines = lines[actions_start:actions_end]
                future_plan = lines[actions_end].strip()
                
                # Parse all actions first
                parsed_actions = []
                for action in action_lines:
                    if '||' in action:
                        tool_name, params_str = action.split('||', 1)
                        tool_name = tool_name.strip()
                        params = {}
                        if params_str.strip() != '{}':
                            try:
                                params = json.loads(params_str)
                            except json.JSONDecodeError:
                                print(f"Error parsing parameters: {params_str}")
                                continue
                        parsed_actions.append({
                            'tool': tool_name,
                            'params': params
                        })
                
                # Execute all tools in parallel
                results = self.execute_tools_parallel(parsed_actions)
                
                # If we got a response (from give_response), return it
                if isinstance(results, str):
                    return results
                
                # If we hit DONE without a give_response, continue the conversation
                if future_plan == "DONE":
                    continue
                    
                # If CONTINUE, loop again to get next batch of actions
                
            except Exception as e:
                print(f"Error processing response: {e}")
                return "Error: Failed to process the response"

def get_hardcoded_vitals(input_text=None):
    print("Getting hardcoded vitals...")
    vitals = {
        "heart_rate": "72 bpm",
        "blood_pressure": "120/80 mmHg",
        "temperature": "98.6°F",
        "respiration_rate": "16 breaths per minute"
    }
    return vitals

def get_system_status(input_text=None):
    print("Getting system status...")
    status = {
        "cpu_usage": "45%",
        "memory_usage": "60%",
        "disk_space": "120GB free out of 256GB",
        "uptime": "5 days, 4 hours, 23 minutes"
    }
    return status

print("Creating agent with tools...")
agent = Agent([
    Tool(
        name="GetHardcodedVitals",
            description="Returns a hardcoded set of vitals: heart rate, blood pressure, temperature, and respiration rate.",
            params=[],
            return_description="Dictionary containing vital signs",
            function=get_hardcoded_vitals
        ),
        Tool(
            name="GetSystemStatus",
            description="Returns a mock system status including CPU usage, memory usage, disk space, and uptime.",
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