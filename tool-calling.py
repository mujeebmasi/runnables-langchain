from dotenv import load_dotenv  
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from rich import print
@tool 
def get_text_length(text: str) -> int:
    """Get the length of the given text."""
    return len(text)

llm = ChatMistralAI(model = "mistral-small")
llm_with_tool = llm.bind_tools([get_text_length])
res = llm_with_tool.invoke("hello ")
print(res)