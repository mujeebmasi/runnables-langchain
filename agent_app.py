from dotenv import load_dotenv
load_dotenv()

import os
import requests
import streamlit as st

from tavily import TavilyClient
from langchain.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
    AIMessage
)

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="City Intelligence Agent",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 City Intelligence Agent")

# ---------------- TOOLS ---------------- #

@tool
def get_weather(city: str) -> str:
    """Fetches the current weather for a given city using the OpenWeatherMap API."""

    api_key = os.getenv("OPEN_WEATHER_API_KEY")

    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={api_key}&units=metric"
    )

    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        weather_desc = data['weather'][0]['description']
        temp = data['main']['temp']

        return (
            f"The current weather in {city} is "
            f"{weather_desc} with a temperature of {temp}°C."
        )

    return "Weather data not found."


tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

@tool
def search_web(city: str) -> str:
    """get latest news in {city}."""

    response = tavily_client.search(
        query=f"Latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    result = response.get("results", [])

    if not result:
        return "No news found."

    news_list = []

    for r in result:
        title = r.get("title", "No title")
        url = r.get("url", "No URL")

        news_list.append(f"{title} - {url}")

    return "\n".join(news_list)

# ---------------- LLM ---------------- #

llm = ChatMistralAI(
   model="open-mistral-7b"
)

tools = {
    "get_weather": get_weather,
    "search_web": search_web
}

llm_with_tools = llm.bind_tools([
    get_weather,
    search_web
])

# ---------------- SESSION STATE ---------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- DISPLAY CHAT ---------------- #

for message in st.session_state.messages:

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

    elif isinstance(message, ToolMessage):
        with st.chat_message("assistant"):
            st.info(message.content)

# ---------------- USER INPUT ---------------- #

user_input = st.chat_input("Ask about weather or city news...")

if user_input:

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    human_message = HumanMessage(content=user_input)

    st.session_state.messages.append(human_message)

    # ---------------- AGENT LOOP ---------------- #

    while True:

        result = llm_with_tools.invoke(
            st.session_state.messages
        )

        st.session_state.messages.append(result)

        if result.tool_calls:

            for tool_call in result.tool_calls:

                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                tool_result = tools[tool_name].invoke(tool_args)

                with st.chat_message("assistant"):
                    st.info(f" Tool Used: {tool_name}")
                    st.write(tool_result)

                tool_message = ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                )

                st.session_state.messages.append(tool_message)

        else:

            with st.chat_message("assistant"):
                st.markdown(result.content)

            break