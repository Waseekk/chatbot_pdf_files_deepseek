from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory

class LLMManager:
    """Manages LLM models and their setup."""
    
    def __init__(self, base_url="http://localhost:11434"):
        """Initialize available models."""
        self.base_url = base_url
        self.setup_models()
    
    def setup_models(self):
        """Initialize available models."""
        self.models = {
            "Deepseek-1.5B": ChatOllama(
                model="deepseek-r1:1.5b",
                temperature=0.3,
                base_url=self.base_url
            ),
            "Deepseek-8B": ChatOllama(
                model="deepseek-r1:8b",
                temperature=0.3,
                base_url=self.base_url
            ),
            "LLaMA3-8B": ChatOllama(
                model="llama3:8b",
                temperature=0.3,
                base_url=self.base_url
            )
        }
        
        # Initialize memory separately for each model
        self.memories = {
            name: ConversationBufferMemory(return_messages=True)
            for name in self.models.keys()
        }
    
    def get_model(self, model_name):
        """Get a specific model by name."""
        return self.models.get(model_name)
    
    def get_memory(self, model_name):
        """Get memory for a specific model."""
        return self.memories.get(model_name)
    
    def get_available_models(self):
        """Return list of available model names."""
        return list(self.models.keys())