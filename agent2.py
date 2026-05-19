from dotenv import load_dotenv
load_dotenv()
import os
import requests
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from rich import print
from tavily import TavilyClient 
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""
    
    api_key = os.getenv("OPEN_WEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
    
    response = requests.get(url)
    data = response.json()
    
    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"
    
    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    
    return f"Weather in {city}: {desc}, {temp}°C"

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

@tool
def search_web(city: str) -> str:
    """Get latest news about a city"""
    
    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )
    
    results = response.get("results", [])
    
    if not results:
        return f"No news found for {city}"
    
    news_list = []
    
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        
        news_list.append(
            f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}..."
        )
    
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before every tool call."""
    tool_name = request.tool_call["name"]
    confirm = input(f"Agent wants to call '{tool_name}'. Approve? (yes/no): ")

    if confirm.lower() != "yes":
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id=request.tool_call["id"]
        )

    return handler(request)  
agent = create_agent(
    llm,
    middleware=[human_approval],
    tools = [get_weather, search_web],
    system_prompt = """
    You are a helpful city assistant.

    Rules:
    - Use tools only when necessary.
    - Call a tool only ONCE per user request.
    - After receiving tool results, answer the user directly.
    - Do not call the same tool repeatedly.
    - Do not loop tool calls.
    """   
    )

print("Type 'exit' to quit.")
print("City Agent: ")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting the agent. Goodbye!")
        break
    response = agent.invoke({
        "messages": [{"role": "user", 
                      "content": user_input}]
    })
    print("Agent:", response["messages"][-1].content)