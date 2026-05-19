from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from rich import print

@tool
def get_text_length(text: str) -> int:
    """Returns the number of characters in a given text"""
    return len(text)

tools = {
    "get_text_length": get_text_length
}

llm = ChatMistralAI(model="mistral-small-2506")

llm_with_tool = llm.bind_tools([get_text_length])

messages = []

prompt = input("You: ")

messages.append(HumanMessage(content=prompt))

# First model call
result = llm_with_tool.invoke(messages)

messages.append(result)

# If tool is needed
if result.tool_calls:

    tool_name = result.tool_calls[0]["name"]

    tool_args = result.tool_calls[0]["args"]

    tool_result = tools[tool_name].invoke(tool_args)

    print(f"\nTool Result: {tool_result}")

    tool_message = ToolMessage(
        content=str(tool_result),
        tool_call_id=result.tool_calls[0]["id"]
    )

    messages.append(tool_message)

    # Final response after tool
    final_result = llm_with_tool.invoke(messages)

    print("\nAI:", final_result.content)

# If no tool needed
else:
    print("\nAI:", result.content)