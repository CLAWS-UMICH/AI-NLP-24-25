import json
import os
from typing import List, Dict
from pathlib import Path

def load_conversation_files(logs_dir: str = "conversation_logs") -> List[Dict]:
    """Load all JSON conversation files from the specified directory."""
    conversations = []
    for file_path in Path(logs_dir).glob("*.json"):
        with open(file_path, 'r') as f:
            try:
                # Skip the system prompt that contains format instructions
                data = json.load(f)
                if data and len(data) > 1:  # If there's more than just the system message
                    conversations.append(data[1:])  # Skip the first system message
            except json.JSONDecodeError:
                print(f"Error loading {file_path}")
    return conversations

def format_message(message: Dict) -> Dict:
    """Convert message format to standardized format."""
    if "from_" in message:
        role = message["from_"]
        if role == "user":
            role = "user"
        elif role == "ai":
            role = "assistant"
        elif role == "system":
            role = "system"
        return {"role": role, "content": message.get("content", "")}
    elif "tool_used" in message:
        # Format tool calls as assistant messages
        tool_name = message.get("tool_used", "")
        params = message.get("params", {})
        return_value = message.get("return_value", "")
        content = f"Function Call: {tool_name}({json.dumps(params)})\nResult: {json.dumps(return_value)}"
        return {"role": "assistant", "content": content}
    return None

def split_into_chains(conversation: List[Dict]) -> List[List[Dict]]:
    """Split a conversation into multiple training chains."""
    chains = []
    
    # Convert messages to standardized format
    formatted_messages = []
    system_message = {"role": "system", "content": "You are a helpful AI assistant."}  # Simple system message
    formatted_messages.append(system_message)
    
    for msg in conversation:
        formatted = format_message(msg)
        if formatted:
            formatted_messages.append(formatted)
    
    # Create the basic chain (just query and first response)
    user_msgs = [m for m in formatted_messages if m["role"] == "user"]
    asst_msgs = [m for m in formatted_messages if m["role"] == "assistant"]
    
    if user_msgs and asst_msgs:
        basic_chain = [system_message, user_msgs[0], asst_msgs[0]]
        chains.append(basic_chain)
    
    # Create the full chain with all messages
    if len(formatted_messages) > 3:  # If there's more than system + user + assistant
        chains.append(formatted_messages)
    
    return chains

def format_for_training(chains: List[List[Dict]]) -> List[Dict]:
    """Format the chains into training data format."""
    training_data = []
    
    for chain in chains:
        conversation = ""
        for message in chain:
            role = message["role"]
            content = message["content"]
            
            # Format in Qwen-2.5 style
            conversation += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        if conversation:
            training_data.append({
                "conversations": chain,
                "text": conversation.strip()
            })
    
    return training_data

def combine_conversation_data(output_file: str = "training_data.json"):
    """Main function to combine and process conversation data."""
    # Load all conversation files
    conversations = load_conversation_files()
    
    # Process all conversations
    all_training_data = []
    for conv in conversations:
        # Split into different chains
        chains = split_into_chains(conv)
        
        # Format chains for training
        training_examples = format_for_training(chains)
        all_training_data.extend(training_examples)
    
    # Save combined training data
    with open(output_file, 'w') as f:
        json.dump(all_training_data, f, indent=2)
    
    print(f"Processed {len(conversations)} conversations into {len(all_training_data)} training examples")

if __name__ == "__main__":
    combine_conversation_data()
