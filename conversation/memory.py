import streamlit as st
from langchain.memory import ConversationBufferMemory
from typing import List, Dict

class ConversationMemory:
    """Manages conversation memory for different LLM models."""
    
    def __init__(self):
        """Initialize memory stores for different models."""
        self.memories = {}
        
    def get_memory(self, model_name: str):
        """Get or create memory for a specific model."""
        if model_name not in self.memories:
            self.memories[model_name] = ConversationBufferMemory(return_messages=True)
        return self.memories[model_name]
    
    def add_to_memory(self, model_name: str, user_input: str, assistant_response: str):
        """Add interaction to the memory for a specific model."""
        memory = self.get_memory(model_name)
        memory.save_context(
            {"input": user_input},
            {"output": assistant_response}
        )
    
    def get_recent_messages(self, model_name: str, count: int = 4):
        """Get the most recent messages from the memory."""
        memory = self.get_memory(model_name)
        messages = memory.chat_memory.messages
        return messages[-count:] if len(messages) >= count else messages
    
    def clear_memory(self, model_name: str = None):
        """Clear memory for a specific model or all models."""
        if model_name:
            if model_name in self.memories:
                self.memories[model_name].clear()
        else:
            for memory in self.memories.values():
                memory.clear()