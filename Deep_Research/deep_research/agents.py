import uuid
import time
import functools
import json
from typing import Any, Callable, Optional, Dict, List
from pydantic import BaseModel
import openai
from openai import OpenAI
import os
import asyncio
import aiohttp
from bs4 import BeautifulSoup

def gen_trace_id() -> str:
    """Generate a unique trace ID"""
    return str(uuid.uuid4())

def trace(func: Callable = None, trace_id: str = None) -> Callable:
    """Decorator to trace function execution"""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            tid = trace_id or gen_trace_id()
            print(f"[TRACE {tid[:8]}] Starting {f.__name__}")
            start_time = time.time()
            
            try:
                result = f(*args, **kwargs)
                end_time = time.time()
                print(f"[TRACE {tid[:8]}] Finished {f.__name__} in {end_time - start_time:.2f}s")
                return result
            except Exception as e:
                end_time = time.time()
                print(f"[TRACE {tid[:8]}] Error in {f.__name__} after {end_time - start_time:.2f}s: {e}")
                raise
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)

class ModelSettings:
    """Model configuration settings"""
    
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.7, 
                 max_tokens: int = 2000, tool_choice: str = "auto", **kwargs):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tool_choice = tool_choice
        
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __str__(self):
        return f"ModelSettings(model={self.model}, temp={self.temperature})"


class WebSearchTool:
    """Web search tool using SerpAPI or direct web scraping"""
    
    def __init__(self, api_key: Optional[str] = None, search_context_size: str = "medium", 
                 max_results: int = 5, **kwargs):
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        self.search_context_size = search_context_size
        self.max_results = max_results
        self.session = None
    
    async def search(self, query: str, num_results: Optional[int] = None) -> List[Dict]:
        """Perform actual web search"""
        num_results = num_results or self.max_results
        print(f"[WebSearchTool] Searching for: {query}")
        
        if self.api_key:
            # Use SerpAPI if available
            return await self._serpapi_search(query, num_results)
        else:
            # Fallback to Google search scraping
            return await self._google_search(query, num_results)
    
    async def _serpapi_search(self, query: str, num_results: int) -> List[Dict]:
        """Search using SerpAPI"""
        async with aiohttp.ClientSession() as session:
            params = {
                "q": query,
                "api_key": self.api_key,
                "num": num_results,
                "engine": "google"
            }
            
            async with session.get("https://serpapi.com/search", params=params) as resp:
                data = await resp.json()
                
                results = []
                for item in data.get("organic_results", [])[:num_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "content": await self._fetch_content(session, item.get("link", ""))
                    })
                
                return results
    
    async def _google_search(self, query: str, num_results: int) -> List[Dict]:
        """Fallback Google search scraping"""
        async with aiohttp.ClientSession() as session:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            params = {"q": query, "num": num_results}
            async with session.get("https://www.google.com/search", params=params, headers=headers) as resp:
                html = await resp.text()
                
            soup = BeautifulSoup(html, "html.parser")
            results = []
            
            for g in soup.find_all('div', class_='g')[:num_results]:
                title_elem = g.find('h3')
                link_elem = g.find('a')
                snippet_elem = g.find('span', class_='aCOpRe')
                
                if title_elem and link_elem:
                    url = link_elem.get('href', '')
                    results.append({
                        "title": title_elem.text,
                        "url": url,
                        "snippet": snippet_elem.text if snippet_elem else "",
                        "content": await self._fetch_content(session, url)
                    })
            
            return results
    
    async def _fetch_content(self, session: aiohttp.ClientSession, url: str) -> str:
        """Fetch and extract text content from URL"""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
            async with session.get(url, headers=headers, timeout=10) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    soup = BeautifulSoup(html, "html.parser")
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.extract()
                    
                    text = soup.get_text()
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    
                    return text[:2000]  # Limit content length
        except Exception as e:
            print(f"Error fetching {url}: {e}")
        
        return ""


class AgentResult:
    """Result wrapper for agent execution"""
    def __init__(self, output: Any, output_type: Optional[type] = None):
        self.final_output = output
        self._output_type = output_type
    
    def final_output_as(self, output_type: type) -> Any:
        """Cast output to specific type"""
        if isinstance(self.final_output, output_type):
            return self.final_output
        elif isinstance(self.final_output, dict) and issubclass(output_type, BaseModel):
            return output_type(**self.final_output)
        elif isinstance(self.final_output, str) and issubclass(output_type, BaseModel):
            try:
                # Try to parse JSON from string
                data = json.loads(self.final_output)
                return output_type(**data)
            except:
                # If that fails, create a default instance
                return output_type()
        return self.final_output


class Agent:
    """OpenAI-powered agent"""
    
    def __init__(self, name: str, instructions: Optional[str] = None, 
                 model: str = "gpt-4o-mini", output_type: Optional[Any] = None,
                 model_settings: Optional[ModelSettings] = None, tools: Optional[list] = None):
        self.name = name
        self.instructions = instructions or "You are a helpful assistant."
        self.model = model
        self.output_type = output_type
        self.model_settings = model_settings or ModelSettings(model=model)
        self.tools = tools or []
        self.trace_id = gen_trace_id()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    @trace
    async def run(self, task: str, context: Optional[dict] = None) -> AgentResult:
        """Run agent task using OpenAI"""ذ
        print(f"[Agent {self.name}] Processing task...")
        
        messages = [
            {"role": "system", "content": self.instructions}
        ]
        
        if context:
            messages.append({"role": "user", "content": f"Context: {json.dumps(context)}"})
        
        messages.append({"role": "user", "content": task})
        
        # If we have tools, use them
        if self.tools and isinstance(self.tools[0], WebSearchTool):
            # Perform searches if needed
            search_results = await self._handle_search(task)
            if search_results:
                messages.append({
                    "role": "system", 
                    "content": f"Search results:\n{json.dumps(search_results, indent=2)}"
                })
        
        # Add output format instruction if needed
        if self.output_type and hasattr(self.output_type, '__fields__'):
            schema = self._get_pydantic_schema(self.output_type)
            messages.append({
                "role": "system",
                "content": f"You must respond with valid JSON matching this schema: {json.dumps(schema)}"
            })
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                temperature=self.model_settings.temperature,
                max_tokens=self.model_settings.max_tokens
            )
            
            result_text = response.choices[0].message.content
            
            # Try to parse as JSON if we have an output type
            if self.output_type and hasattr(self.output_type, '__fields__'):
                try:
                    # Clean up the response to extract JSON
                    json_start = result_text.find('{')
                    json_end = result_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = result_text[json_start:json_end]
                        result_data = json.loads(json_str)
                        return AgentResult(result_data, self.output_type)
                except:
                    pass
            
            return AgentResult(result_text, self.output_type)
            
        except Exception as e:
            print(f"Error in agent {self.name}: {e}")
            raise
    
    async def _handle_search(self, query: str) -> List[Dict]:
        """Handle search if we have search tools"""
        for tool in self.tools:
            if isinstance(tool, WebSearchTool):
                # Extract search query from the task
                search_query = query.split("Search term:")[-1].split("\n")[0].strip()
                if not search_query:
                    search_query = query[:100]
                
                results = await tool.search(search_query, num_results=3)
                return results
        return []
    
    def _get_pydantic_schema(self, model_class) -> dict:
        """Get JSON schema from Pydantic model"""
        schema = {"type": "object", "properties": {}, "required": []}
        
        for field_name, field_info in model_class.__fields__.items():
            field_type = field_info.annotation
            
            # Handle basic types
            if field_type == str:
                schema["properties"][field_name] = {"type": "string"}
            elif field_type == int:
                schema["properties"][field_name] = {"type": "integer"}
            elif field_type == float:
                schema["properties"][field_name] = {"type": "number"}
            elif field_type == bool:
                schema["properties"][field_name] = {"type": "boolean"}
            elif hasattr(field_type, '__origin__') and field_type.__origin__ == list:
                # Handle List types
                schema["properties"][field_name] = {"type": "array", "items": {"type": "string"}}
            else:
                schema["properties"][field_name] = {"type": "object"}
            
            if field_info.is_required():
                schema["required"].append(field_name)
        
        return schema


class Runner:
    """Runner class for executing agent tasks"""
    
    @staticmethod
    async def run(agent: Agent, task: str, *args, **kwargs) -> AgentResult:
        """Run an agent task"""
        return await agent.run(task, *args, **kwargs)