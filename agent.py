import os
import math
from typing import TypedDict, Optional, List, Any
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pydantic import SecretStr
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Any]

# Global LLM instance
_llm = None

def get_llm():
    
    """Get or create LLM instance - prefer Groq if available, fallback to Gemini."""
    print("Getting LLM")
    global _llm
    if _llm is None:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            _llm = ChatGroq(model="qwen-qwq-32b", temperature=0, api_key=SecretStr(groq_api_key))
        else:
            _llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
    return _llm

@tool
def web_search(query: str) -> str:
    
    """Search the web for current information, news, and general queries.
    Args:
        query: The query to search the web for.
    Returns:
        A string containing the search results.
    """
    print("Web search tool called")
    try:
        tavily = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"), max_results=3)
        docs = tavily.invoke(query)
        
        if not docs:
            return "No search results found."
        
        results = []
        for i, doc in enumerate(docs[:3], 1):
            content = doc.get('content', '')[:600]  # Limit content length
            title = doc.get('title', 'No title')
            url = doc.get('url', '')
            results.append(f"{i}. {title}\nURL: {url}\nContent: {content}")
        
        return f"Search results for '{query}':\n\n" + "\n\n".join(results)
        
    except Exception as e:
        return f"Search error: {e}"

@tool
def wikipedia_search(query: str) -> str:
    
    """Search Wikipedia for encyclopedic information about people, places, events, etc.
    Args:
        query: The query to search Wikipedia for.
    Returns:
        A string containing the search results.
    """
    print("Wikipedia tool called")
    try:
        loader = WikipediaLoader(query=query, load_max_docs=2)
        docs = loader.load()
        
        if not docs:
            return f"No Wikipedia articles found for: {query}"
        
        results = []
        for doc in docs:
            title = doc.metadata.get('title', 'Unknown')
            content = doc.page_content[:800]  # Limit content
            results.append(f"Wikipedia Article: {title}\nContent: {content}")
        
        return f"Wikipedia results for '{query}':\n\n" + "\n\n---\n\n".join(results)
        
    except Exception as e:
        return f"Wikipedia search error: {e}"

@tool
def calculator(expression: str) -> str:
   
    """Perform mathematical calculations. Supports basic operations and math functions.
    Args:
        expression: The mathematical expression to evaluate.
    Returns:
        A string containing the calculation result.
    """
    print("Calculator tool called")
    try:
        # Safe evaluation with math functions
        safe_dict = {
            "__builtins__": {},
            "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow,
            "sin": math.sin, "cos": math.cos, "tan": math.tan, "sqrt": math.sqrt,
            "log": math.log, "log10": math.log10, "exp": math.exp, "pi": math.pi, "e": math.e,
            "floor": math.floor, "ceil": math.ceil
        }
        result = eval(expression, safe_dict)
        return f"Calculation result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"

@tool
def arxiv_search(query: str) -> str:
    
    """Search ArXiv for academic papers and research articles.
    Args:
        query: The query to search ArXiv for.
    Returns:
        A string containing the search results.
    """
    print("ArXiv tool called")
    try:
        arxiv_wrapper = ArxivAPIWrapper(
            arxiv_search=query,
            top_k_results=2,
            load_max_docs=2,
            doc_content_chars_max=800,
            arxiv_exceptions=Any
        )
        
        arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
        result = arxiv_tool.run(query)
        
        if not result or result.strip() == "":
            return f"No ArXiv papers found for: {query}"
        else:
            return f"ArXiv results for '{query}':\n\n{result}"
    
    except Exception as e:
        return f"ArXiv search error: {e}"

# Define all tools
tools = [web_search, wikipedia_search, calculator, arxiv_search]

def model_node(state: AgentState) -> AgentState:
    
    """Main model node that processes queries and calls tools when needed.
    Args:
        state: The current state of the agent.
    Returns:
        The updated state of the agent.
    """
    print("Model node called")
    messages = state["messages"]
    
    # Get LLM with tools bound
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)
    
    # Add system message if not present
    if not messages or not isinstance(messages[0], AIMessage):
        system_message = AIMessage(content="""You are a helpful assistant tasked with answering questions using a set of tools.

Your final answer must strictly follow this format:
FINAL ANSWER: [ANSWER]

Only write the answer in that exact format. Do not explain anything. Do not include any other text.


Only use tools if the current question is different from the similar one.

Examples:
- FINAL ANSWER: FunkMonk
- FINAL ANSWER: Paris
- FINAL ANSWER: 128

If you do not follow this format exactly, your response will be considered incorrect.""")
        messages = [system_message] + messages
    
    try:
        response = llm_with_tools.invoke(messages)
        return {"messages": messages + [response]}
    except Exception as e:
        error_msg = AIMessage(content=f"Error: {e}")
        return {"messages": messages + [error_msg]}

def should_continue(state: AgentState) -> str:
    
    """Determine if we should continue with tool calls or end.
    Args:
        state: The current state of the agent.
    Returns:
        A string indicating whether to continue with tool calls or end.
    """
    print("Should continue called")
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message has tool calls, go to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Otherwise, we're done
    return "end"

def build_graph():
    
    """Build and compile the agent graph."""
    print("Building graph")
    # Create tool node
    tool_node = ToolNode(tools)
    
    # Create graph
    builder = StateGraph(AgentState)
    
    # Add nodes
    builder.add_node("agent", model_node)
    builder.add_node("tools", tool_node)
    
    # Add edges
    builder.add_edge(START, "agent")
    
    # Conditional edge from agent
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # Edge from tools back to agent
    builder.add_edge("tools", "agent")
    
    return builder.compile()

def run_agent(query: str) -> str:

    """Run the agent with a query and return the final answer.
    Args:
        query: The query to run the agent with.
    Returns:
        A string containing the final answer.
    """
    print("Running agent")
    graph = build_graph()
    
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    try:
        # Run the graph
        final_state = graph.invoke(initial_state)
        
        # Extract the final answer from the last AI message
        messages = final_state["messages"]
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not hasattr(message, 'tool_calls'):
                return str(message.content)
            elif isinstance(message, AIMessage) and hasattr(message, 'tool_calls') and not message.tool_calls:
                return str(message.content)
        
        return "No answer generated"
        
    except Exception as e:
        return f"Error: {e}"
