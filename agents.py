from dotenv import load_dotenv
load_dotenv()
import os
import requests
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from rich import print
from tavily import TavilyClient 
@tool
def get_weather(city: str) -> str:
    """Fetches the current weather for a given city using the OpenWeatherMap API."""
    api_key = os.getenv("OPEN_WEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json() 
    if response.status_code == 200:
        weather_desc = data['weather'][0]['description']
        temp = data['main']['temp']
        return f"The current weather in {city} is {weather_desc} with a temperature of {temp}°C."
res = get_weather.invoke("New York")
print(res)

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

@tool
def search_web(city: str) -> str:
    """get latest news in {city}."""

    response = tavily_client.search(query=f"Latest news in {city}",
                                    search_depth="basic", 
                                    max_results=3)

    result = response.get("results", [])
    if not result:
        return "No news found."
    news_list = []
    for r in result:
        title = r.get("title", "No title")
        url = r.get("url", "No URL")
        news_list.append(f"{title} - {url}")
    return "\n".join(news_list)

res2 = search_web.invoke("New York")
print(res2)

llm = ChatMistralAI(model="mistral-small")

tools = {
    "get_weather": get_weather,
    "search_web": search_web
    }

llm_with_tools = llm.bind_tools([get_weather, search_web])

#Agent Loop

messages = []
print("City Intelligence Agent")
print("Type 'exit' to quit. ")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    messages.append(HumanMessage(content=user_input))
    
    while True:
        result = llm_with_tools.invoke(messages)
        messages.append(result)

        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_result = tools[tool_name].invoke(tool_args)
                print(f"\nTool Result: {tool_result}")
                tool_message = ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                )
                messages.append(tool_message)
            
        else:
            print("\nAI:", result.content)
            break