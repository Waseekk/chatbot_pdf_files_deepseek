from langchain.memory import ConversationBufferMemory
from typing import List, Dict
import json
from datetime import datetime
import pandas as pd

class ConversationManager:
    """Manages conversation history and memory."""
    
    def __init__(self):
        self.memory = ConversationBufferMemory(return_messages=True)
        self.conversation_logs = []
    
    def add_interaction(self, user_input: str, assistant_response: str, references: List[Dict]):
        """Add an interaction to the conversation history."""
        timestamp = datetime.now().isoformat()
        interaction = {
            'timestamp': timestamp,
            'user_input': user_input,
            'assistant_response': assistant_response,
            'references': references
        }
        self.conversation_logs.append(interaction)
        
        # Update memory
        self.memory.save_context(
            {"input": user_input},
            {"output": assistant_response}
        )
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the recent conversation history."""
        messages = self.memory.chat_memory.messages[-5:]  # Last 5 messages
        summary = []
        for msg in messages:
            summary.append(f"{msg.type}: {msg.content[:100]}...")
        return "\n".join(summary)
    
    def export_conversation(self, format: str = 'json') -> str:
        """Export conversation history in specified format."""
        if format == 'json':
            return json.dumps(self.conversation_logs, indent=2)
        elif format == 'csv':
            df = pd.DataFrame(self.conversation_logs)
            return df.to_csv(index=False)
        return ""