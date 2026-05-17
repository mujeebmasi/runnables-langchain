from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple terms."
)
 
model = ChatMistralAI(model = "mistral-small")

parser = StrOutputParser()

# formatted_prompt = prompt.format(topic="quantum computing")

# response = model.invoke(formatted_prompt)

# final_output = parser.parse(response)

chain = prompt | model | parser 

result = chain.invoke({"topic": "Machine Learning"})  
print(result)