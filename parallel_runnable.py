from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines."
)
 
detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail."
)
model = ChatMistralAI(model = "mistral-small")

parser = StrOutputParser()

# formatted_prompt = prompt.format(topic="quantum computing")
# response = model.invoke(formatted_prompt)
# final_output = parser.parse(response)



chain = RunnableParallel({
    "short":RunnableLambda(lambda x:x["short"])| short_prompt | model | parser, 
    "detailed":RunnableLambda(lambda x:x["detailed"])| detailed_prompt | model | parser
})

result = chain.invoke({
    "short":{"topic": "Machine Learning"},
    "detailed":{"topic": "Langchain"}
})

#chain uses RunnableParallel and RunnableLamda
#RunnableParallel to work 2 pipelines parallelly
#RunnableLambda to give different inputs to each pipepline

chain2 = RunnableParallel({
    "short": short_prompt | model | parser,
    "detailed": detailed_prompt | model | parser
})

result2 = chain2.invoke({
    "topic":"Aerodynamics"
})

#chain2 just uses RunnableParallel
#which means both pipelines will use same input but generate different outputs based on the prompts

print(result["short"])   
print("----------------------------------------------------------------------------------")   
print(result["detailed"])
print("----------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------")
print(result2["short"])
print("----------------------------------------------------------------------------------")
print(result2["detailed"])