from dotenv import load_dotenv
load_dotenv()
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

search_tool = TavilySearchResults(max_result = 5)

model = ChatMistralAI(model = "mistral-small")
prompt = ChatPromptTemplate.from_template(
    "What are the latest news about {topic}?"
)
chain = prompt |  model | StrOutputParser()
news_result = search_tool.run("Latest news about AI")
result = chain.invoke({"topic": news_result})
print(result)
