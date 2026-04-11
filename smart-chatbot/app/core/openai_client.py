import openai
from openai import AsyncOpenAI
from typing import List, Dict, Optional
from ..core.config import settings

class OpenAIClient:
    def __init__(self):
        # فقط AsyncOpenAI را مقداردهی می‌کنیم
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.default_model = settings.openai_model
        self.default_temperature = settings.temperature
        self.default_max_tokens = settings.max_tokens
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ) -> Dict:
        """Generate response using OpenAI API with prompt engineering"""
        
        # Apply prompt engineering techniques
        enhanced_messages = self._enhance_prompts(messages)
        
        response = await self.client.chat.completions.create(
            model=model or self.default_model,
            messages=enhanced_messages,
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens,
            presence_penalty=0.6,
            frequency_penalty=0.3
        )
        
        return {
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    
    def _enhance_prompts(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Apply prompt engineering best practices"""
        
        # Add system message if not present
        if not messages or messages[0].get("role") != "system":
            system_prompt = {
                "role": "system",
                "content": """You are a helpful, intelligent assistant. Follow these guidelines:
1. Provide clear, concise, and accurate responses
2. Ask for clarification when the query is ambiguous
3. Use structured formatting when appropriate
4. Be friendly and professional
5. Admit when you don't know something
6. Provide sources or references when making factual claims"""
            }
            messages.insert(0, system_prompt)
        
        return messages

# یک instance که در سرتاسر اپ استفاده می‌کنیم
openai_client = OpenAIClient()
