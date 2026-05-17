from langchain.tools import tool
@tool
def get_greeting(name: str) -> str:
    """Get a greeting message for the given name."""
    return f"Hello {name}, Welcome!"
     
result = get_greeting.invoke({"name":"Mujeeb"})
print(result)

