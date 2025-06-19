from re import search
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional, Any, Dict, Literal
from langchain_core.runnables import RunnableLambda
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.tools import Tool
from langchain_core.documents import Document
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.tools import tool
import yt_dlp
import requests
import json
import os
import math
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class State(TypedDict):
    query: str
    tool_call: Optional[Dict[str, Any]]
    tool_output: Optional[str]
    answer: Optional[str]

def build_graph():
    """Build and return the compiled graph."""
    builder = StateGraph(State)

    # Add all nodes
    builder.add_node("model", model_node)
    builder.add_node("search", search_node)
    builder.add_node("browse", browse_node)  
    builder.add_node("youtube", youtube_node)
    builder.add_node("python", python_node)
    builder.add_node("calculator", calculator_node)
    builder.add_node("post_process", post_process_node)
    
    # Add conditional edges from model to tools
    builder.add_conditional_edges(
        "model",
        route_to_tool,
        {
            "search": "search",
            "browse": "browse", 
            "youtube": "youtube",
            "python": "python",
            "calculator": "calculator",
            "post_process": "post_process"
        }
    )
    
    # All tool nodes go to post_process
    builder.add_edge("search", "post_process")
    builder.add_edge("browse", "post_process")
    builder.add_edge("youtube", "post_process")
    builder.add_edge("python", "post_process")
    builder.add_edge("calculator", "post_process")
    
    # Post process goes to END
    builder.add_edge("post_process", END)
    
    # Start with model
    builder.add_edge(START, "model")

    return builder.compile()

def route_to_tool(state: State) -> Literal["search", "browse", "youtube", "python", "calculator", "post_process"]:
    """Route to the appropriate tool based on the tool_call in state."""
    tool_call = state.get("tool_call")
    if not tool_call or tool_call is None:
        return "post_process"
    
    tool_name = tool_call.get("tool") if isinstance(tool_call, dict) else None
    if tool_name == "search_tool":
        return "search"
    elif tool_name == "browse_tool":
        return "browse"
    elif tool_name == "youtube_tool":
        return "youtube"
    elif tool_name == "python_tool":
        return "python"
    elif tool_name == "calculator_tool":
        return "calculator"
    else:
        return "post_process"

@tool
def calculator_tool(query: str) -> str:
    """
    This function is used to calculate a mathematical expression.
    Args:
        query (str): The mathematical expression to evaluate.
    Returns:
        str: The result of the evaluation or an error message.
    """
    try:
        # Create a safe namespace for evaluation
        safe_dict = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "math": math,
            # Add common math functions directly
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "sqrt": math.sqrt,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e
        }
        
        result = eval(query, safe_dict)
        result_str = str(result)
    except Exception as e:
        result_str = f"Error evaluating expression: {e}"
    return result_str

@tool
def search_tool(query: str) -> str:
    """
    Search the web for the given query using Tavily API.
    Args:
        query: The search query string.
    Returns:
        A formatted string of search results.
    """
    try:
        tavily = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"), max_results=3)
        docs = tavily.invoke(query)
        
        if not docs:
            return "No search results found."
            
        formatted_docs = []
        for doc in docs:
            source = doc.get('url', 'Unknown source')
            content = doc.get('content', 'No content available')
            formatted_docs.append(f'<Document source="{source}"/>\n{content}\n</Document>')
            
        return "\n\n---\n\n".join(formatted_docs)
    except Exception as e:
        return f"Error during Tavily search: {e}"

@tool
def browse_tool(query: str) -> str:
    """
    Search Wikipedia and return summarized results.
    Args:
        query (str): The search query.
    Returns:
        str: Formatted search results or error message.
    """
    try:
        loader = WikipediaLoader(query=query, load_max_docs=2)
        docs = loader.load()
        
        if not docs:
            return f"No Wikipedia articles found for: {query}"
            
        formatted_docs = []
        for doc in docs:
            source = doc.metadata.get('source', 'Wikipedia')
            title = doc.metadata.get('title', 'Unknown title')
            content = doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content
            formatted_docs.append(f'<Document source="{source}" title="{title}"/>\n{content}\n</Document>')
            
        return "\n\n---\n\n".join(formatted_docs)
    except Exception as e:
        return f"Error during Wikipedia search: {e}"

@tool
def python_tool(query: str) -> str:
    """
    Execute a Python code snippet and return the output or error.
    Args:
        query (str): Python code to execute.
    Returns:
        str: Output from the execution or error message.
    """
    try:
        repl = PythonREPLTool()
        result = repl.invoke(query)
        return str(result)
    except Exception as e:
        return f"Python execution error: {e}"

@tool
def youtube_tool(query: str) -> str:
    """
    Get the transcript text from a YouTube video URL.
    Args:
        query (str): The YouTube video URL.
    Returns:
        str: Transcript text or an error message.
    """
    try:
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)
            
            if not info:
                return "Failed to extract video information."

            # Try automatic captions first, then manual subtitles
            subtitles = info.get('automatic_captions', {}) or info.get('subtitles', {})
            
            if 'en' not in subtitles:
                return "No English subtitles available."

            sub_info = subtitles['en'][0]
            sub_url = sub_info.get('url')
            
            if not sub_url:
                return "Subtitle URL not found."

            response = requests.get(sub_url, timeout=30)
            response.raise_for_status()
            
            # Basic transcript cleaning (remove XML tags if present)
            transcript = response.text
            if '<' in transcript and '>' in transcript:
                import re
                transcript = re.sub(r'<[^>]+>', '', transcript)
                
            return transcript[:2000] + "..." if len(transcript) > 2000 else transcript
            
    except Exception as e:
        return f"Error retrieving YouTube transcript: {e}"

def model_node(state: State) -> State:
    """
    Analyze the query and decide which tool to use.
    """
    query = state.get("query", "")
    if not query:
        return {**state, "tool_call": {"tool": "none", "args": {}}, "answer": "No query provided"}
    
    llm = ChatGroq(model="qwen-qwq-32b", temperature=0)
    
    prompt = f"""
You are an intelligent assistant that decides which tool to use based on the user's query.
Available tools:
- search_tool: For searching the web for recent information, news, current events
- browse_tool: For searching Wikipedia for encyclopedic information, facts, definitions, historical information
- youtube_tool: For extracting transcripts from YouTube videos (only when YouTube URL is provided)
- python_tool: For executing complex Python code, data analysis, programming tasks
- calculator_tool: For simple mathematical calculations and expressions
- none: ONLY if the query is a greeting, thank you, or completely unrelated to information/calculation needs

IMPORTANT: For factual questions like "What is the capital of France?", "Who invented the telephone?", "Tell me about World War 2", etc., use browse_tool to get Wikipedia information.

For recent news, current events, or time-sensitive information, use search_tool.

Query: {query}

Return ONLY a JSON object with 'tool' and 'args'. Examples:

- "What is the capital of France?" -> {{"tool": "browse_tool", "args": {{"query": "France capital Paris"}}}}
- "Tell me about Python programming" -> {{"tool": "browse_tool", "args": {{"query": "Python programming language"}}}}
- "What's the latest news about AI?" -> {{"tool": "search_tool", "args": {{"query": "latest AI news artificial intelligence"}}}}
- "Calculate 15 * 23 + 7" -> {{"tool": "calculator_tool", "args": {{"query": "15 * 23 + 7"}}}}
- "Write code to sort a list" -> {{"tool": "python_tool", "args": {{"query": "# Sort a list\\nmy_list = [3, 1, 4, 1, 5]\\nprint(sorted(my_list))"}}}}
- "https://youtube.com/watch?v=abc" -> {{"tool": "youtube_tool", "args": {{"query": "https://youtube.com/watch?v=abc"}}}}
- "Hello" -> {{"tool": "none", "args": {{}}}}
"""

    try:
        response = llm.invoke(prompt)
        # Handle different response types safely
        if hasattr(response, 'content'):
            if isinstance(response.content, str):
                response_text = response.content
            elif isinstance(response.content, list):
                # Handle list content - join if it's a list of strings
                response_text = ' '.join(str(item) for item in response.content)
            else:
                response_text = str(response.content)
        else:
            response_text = str(response)
        
        # Ensure response_text is a string before regex
        if not isinstance(response_text, str):
            response_text = str(response_text)
        
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            decision = json.loads(json_match.group())
        else:
            # Default to browse_tool for factual questions
            decision = {"tool": "browse_tool", "args": {"query": query}}
            
    except Exception as e:
        print(f"Error in model_node: {e}")
        # Default to browse_tool instead of none
        decision = {"tool": "browse_tool", "args": {"query": query}}
    
    # Ensure decision is a valid dict
    if not isinstance(decision, dict):
        decision = {"tool": "browse_tool", "args": {"query": query}}
    
    # Ensure args is a dict
    if "args" not in decision or not isinstance(decision.get("args"), dict):
        decision["args"] = {"query": query}
    
    # Update state with tool decision - create new state dict
    new_state: State = {
        "query": state.get("query", ""),
        "tool_call": decision,
        "tool_output": state.get("tool_output"),
        "answer": state.get("answer")
    }
    
    # If no tool needed, provide direct answer for greetings only
    if decision.get("tool") == "none":
        new_state["answer"] = "Hello! I'm here to help you with information, calculations, searches, and more. What would you like to know?"
    
    return new_state

# Node functions that wrap the tools
def search_node(state: State) -> State:
    """Execute search tool and update state."""
    tool_call = state.get("tool_call", {})
    args = tool_call.get("args", {}) if tool_call else {}
    query = args.get("query", "") if args else ""
    result = search_tool.invoke(query) if query else "No search query provided"
    
    new_state: State = {
        "query": state.get("query", ""),
        "tool_call": state.get("tool_call"),
        "tool_output": result,
        "answer": state.get("answer")
    }
    return new_state

def browse_node(state: State) -> State:
    """Execute browse tool and update state."""
    tool_call = state.get("tool_call", {})
    args = tool_call.get("args", {}) if tool_call else {}
    query = args.get("query", "") if args else ""
    result = browse_tool.invoke(query) if query else "No browse query provided"
    
    new_state: State = {
        "query": state.get("query", ""),
        "tool_call": state.get("tool_call"),
        "tool_output": result,
        "answer": state.get("answer")
    }
    return new_state

def youtube_node(state: State) -> State:
    """Execute YouTube tool and update state."""
    tool_call = state.get("tool_call", {})
    args = tool_call.get("args", {}) if tool_call else {}
    query = args.get("query", "") if args else ""
    result = youtube_tool.invoke(query) if query else "No YouTube URL provided"
    
    new_state: State = {
        "query": state.get("query", ""),
        "tool_call": state.get("tool_call"),
        "tool_output": result,
        "answer": state.get("answer")
    }
    return new_state

def python_node(state: State) -> State:
    """Execute Python tool and update state."""
    tool_call = state.get("tool_call", {})
    args = tool_call.get("args", {}) if tool_call else {}
    query = args.get("query", "") if args else ""
    result = python_tool.invoke(query) if query else "No Python code provided"
    
    new_state: State = {
        "query": state.get("query", ""),
        "tool_call": state.get("tool_call"),
        "tool_output": result,
        "answer": state.get("answer")
    }
    return new_state

def calculator_node(state: State) -> State:
    """Execute calculator tool and update state."""
    tool_call = state.get("tool_call", {})
    args = tool_call.get("args", {}) if tool_call else {}
    query = args.get("query", "") if args else ""
    result = calculator_tool.invoke(query) if query else "No calculation provided"
    
    new_state: State = {
        "query": state.get("query", ""),
        "tool_call": state.get("tool_call"),
        "tool_output": result,
        "answer": state.get("answer")
    }
    return new_state

def post_process_node(state: State) -> State:
    """
    Post-process the results and generate final answer.
    """
    query = state.get("query", "")
    tool_output = state.get("tool_output")
    existing_answer = state.get("answer")
    
    # If we already have an answer (from direct response), use it
    if existing_answer and not tool_output:
        return state
    
    answer_str = ""
    if tool_output:
        # Generate final answer using LLM with tool output
        llm = ChatGroq(model="qwen-qwq-32b", temperature=0)
        
        prompt = f"""
Based on the user's query and the tool output, provide a helpful and concise answer.

User Query: {query}

Tool Output: {tool_output}

Please provide a clear, helpful response based on this information. Be direct and informative:
"""
        
        try:
            response = llm.invoke(prompt)
            # Handle different response types safely
            if hasattr(response, 'content'):
                if isinstance(response.content, str):
                    answer_str = response.content
                elif isinstance(response.content, list):
                    answer_str = ' '.join(str(item) for item in response.content)
                else:
                    answer_str = str(response.content)
            else:
                answer_str = str(response)
        except Exception as e:
            answer_str = f"Here's what I found: {tool_output}"
    else:
        # Use existing answer or provide default
        answer_str = existing_answer or "I couldn't process your request. Please try again."
    
    # Ensure answer is a string
    if not isinstance(answer_str, str):
        answer_str = str(answer_str)
    
    new_state: State = {
        "query": state.get("query", ""),
        "tool_call": state.get("tool_call"),
        "tool_output": state.get("tool_output"),
        "answer": answer_str
    }
    return new_state

def run_agent(query: str) -> str:
    """
    Run the agent with a query and return the final answer.
    """
    graph = build_graph()
    
    initial_state = {
        "query": query,
        "tool_call": None,
        "tool_output": None,
        "answer": None
    }
    
    try:
        result = graph.invoke(initial_state)
        return result.get("answer", "No answer generated")
    except Exception as e:
        return f"Error running agent: {e}"

if __name__ == "__main__":
    # Test the agent
    test_queries = [
        "What is the capital of France?",
        "Calculate 15 * 23 + 7",
        "Search for recent news about artificial intelligence",
        "Tell me about machine learning",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("="*50)
        result = run_agent(query)
        print(f"Answer: {result}")
        print("-"*50)