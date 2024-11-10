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
        self.tools = {tool.name: tool for tool in tools}
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        
        # Add built-in give_response tool
        self.tools["give_response"] = Tool(
            name="give_response",
            description="Return a response to the user",
            params=[Param("response", "str", "Response text to return to user", True)],
            return_description="None",
            function=lambda response: None
        )

        # Initialize conversation history with system prompt
        self.conversation_history = []
        system_prompt = """You are an AI assistant that helps users by calling appropriate tools.
        
        You MUST respond with a JSON object in this exact format:
        {
            "reasoning": "explanation of why you're calling this tool",
            "action": "name of the tool to use",
            "params": {},
            "futurePlan": "if using a tool other than give_response, explain what you'll do with the result. If using give_response, set this to 'end conversation'"
        }
        
        If the exact format is not followed, AWFUL THINGS WILL HAPPEN.

        Example:
        {
            "reasoning": "Returning a response to the user",
            "action": "give_response",
            "params": {"response": "Hello!"},
            "futurePlan": "end conversation"
        }

        IMPORTANT WORKFLOW:
        1. First, use any necessary tools to gather information
        2. Once you have all needed information, you MUST use give_response to provide the final answer
        3. Never call the same tool multiple times with the same parameters
        4. The conversation ends when you call give_response
        
        Available tools:
        """ + json.dumps({name: {
            'description': tool.description,
            'params': [{'name': p.name, 'type': p.type, 'description': p.description} for p in tool.params]
        } for name, tool in self.tools.items()}, indent=2)

        self.conversation_history.append(Message(from_="system", content=system_prompt))

    def format_conversation_history(self):
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
        
        return formatted_messages

    def execute_tool(self, tool_name, params):
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool = self.tools[tool_name]
        
        # Handle give_response params
        if tool_name == "give_response":
            if isinstance(params, str):
                params = {"response": params}
            elif isinstance(params, dict) and "response" not in params:
                params = {"response": str(params)}
        
        result = tool.function(**params)
        
        self.conversation_history.append(FunctionCall(
            tool_used=tool_name,
            params=params,
            return_value=result
        ))
        
        return result

    def ask(self, user_input):
        self.conversation_history.append(Message(from_="user", content=user_input))
        
        while True:
            messages = self.format_conversation_history()
            
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="gpt-4o-mini",
                temperature=0.2,
                response_format={"type": "json_object"},
                max_tokens=500
            )

            response = chat_completion.choices[0].message.content
            parsed_response = json.loads(response)
            
            # Validate required fields
            required_fields = ["action", "params", "reasoning", "futurePlan"]
            if not all(field in parsed_response for field in required_fields):
                raise ValueError(f"Missing required fields. Response was: {response}")

            print(response)

            # Add AI's message to conversation history
            self.conversation_history.append(Message(
                from_="ai",
                content=response
            ))
            
            # Execute the chosen tool
            result = self.execute_tool(parsed_response["action"], parsed_response["params"])
            
            # If the tool was give_response, we're done
            if parsed_response["action"] == "give_response":
                return parsed_response["params"]["response"]
            
            # Add the tool's result to the conversation history
            self.conversation_history.append(Message(
                from_="system",
                content=f"Tool '{parsed_response['action']}' returned: {result}"
            ))
                    

def calculate_sum(a: int, b: int) -> int:
    return a + b

def get_weather(city: str) -> str:
    # Simplified weather simulation
    return f"The weather in {city} is sunny and 72°F"

def main():
    # Create agent with tools
    agent = Agent([
        Tool(
            name="calculate_sum",
            description="Calculate the sum of two numbers",
            params=[
                Param("a", "int", "First number", True),
                Param("b", "int", "Second number", True)
            ],
            return_description="Sum of the two numbers",
            function=calculate_sum
        ),
        Tool(
            name="get_weather",
            description="Get the weather for a city",
            params=[
                Param("city", "str", "Name of the city", True)
            ],
            return_description="Weather description",
            function=get_weather
        )
    ])
    
    print("AI Assistant ready! Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            break
        response = agent.ask(user_input)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()
