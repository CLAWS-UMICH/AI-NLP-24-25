from dataclasses import dataclass
import json
from openai import OpenAI
import time
import os
from dotenv import load_dotenv

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

        TOOL_NAME_OPTIONS = [tool.name for tool in self.tools.values()]

        system_prompt = """You are an AI assistant. Use tools to help users.
        
        YOU MUST RESPOND WITH EXACTLY 4 LINES, NO EXCEPTIONS:
        Line 1: "PLAN - " followed by your plan
        Line 2: tool name (must be exact match from list below)
        Line 3: parameters (write 'none' if no parameters)
        Line 4: "DONE" or "CONTINUE"
        
        CRITICAL: 
        - ALWAYS include all 4 lines
        - NEVER skip the PLAN line
        - NEVER add extra lines
        - ALWAYS use give_response tool to communicate with user
        
        Example correct response:
        PLAN - I will check the system status
        GetSystemStatus
        none
        CONTINUE
        
        Example response with give_response:
        PLAN - I will tell the user their vitals
        give_response
        response=Your heart rate is 72 bpm
        DONE
        
        Available tools:
        """ + "\n".join(f"- {name}: {tool.description}" for name, tool in self.tools.items())

        self.conversation_history.append(Message(from_="system", content=system_prompt))
        print("System prompt added to conversation history")
    
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
        
        result = tool.function(**params)
        print(f"Tool returned: {result}")
        
        self.conversation_history.append(FunctionCall(
            tool_used=tool_name,
            params=params,
            return_value=result
        ))
        print("Tool execution recorded in conversation history")
        
        return result

    def ask(self, user_input):
        print(f"\nReceived user input: {user_input}")
        self.conversation_history.append(Message(from_="user", content=user_input))
        
        while True:
            print("\nFormatting conversation history for API call...")
            messages = self.format_conversation_history()
            
            print("Calling OpenAI API...")
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=500
            )

            response = chat_completion.choices[0].message.content
            print(f"Received API response:\n{response}")
            
            lines = response.strip().split('\n')
            
            parsed_response = {
                "plan": lines[0].strip(),
                "action": lines[1].strip(),
                "params": dict(p.split('=', 1) for p in lines[2].strip().split('\n') if '=' in p),
                "futurePlan": lines[3].strip()
            }
            print(f"Parsed response: {parsed_response}")

            # Add AI's message to conversation history
            self.conversation_history.append(Message(
                from_="ai",
                content=response
            ))
            print("Added AI response to conversation history")
            
            # Execute the chosen tool
            result = self.execute_tool(parsed_response["action"], parsed_response["params"])
            
            # If the tool was give_response, we're done
            if parsed_response["action"] == "give_response":
                print("give_response tool called, ending conversation turn")
                return parsed_response["params"]["response"]
            
            # Add the tool's result to the conversation history
            self.conversation_history.append(Message(
                from_="system",
                content=f"Tool '{parsed_response['action']}' returned: {result}"
            ))
            print("Added tool result to conversation history")
                    

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