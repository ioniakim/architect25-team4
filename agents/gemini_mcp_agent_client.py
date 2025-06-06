import asyncio
from typing import List, Optional

# New imports for creating a StructuredTool
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from langchain_core.messages import SystemMessage, HumanMessage 
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI


async def create_mcp_react_agent(config: dict, llm):
    """
    Creates a ReAct agent using its default, built-in prompt.
    """
    name = config["name"]
    mcp_config = config["mcp"]

    client = MultiServerMCPClient({
        name: {
            "url": mcp_config["url"],
            "transport": mcp_config.get("transport", "streamable-http")
        },
    })

    tools = await client.get_tools()

    print(f"Successfully fetched {len(tools)} tools for agent: {name}")
    for tool in tools:
        print(f"\t- Tool: {tool.name}, Description: {tool.description}")

    agent_executor = create_react_agent(
        model=llm,
        tools=tools,
    )
    
    return agent_executor


def create_agent_as_tool(agent_executor, system_prompt: str, tool_name: str, tool_description: str) -> StructuredTool:
    """
    Wraps a LangGraph agent executor in a StructuredTool.
    """
    # Define the input schema
    class SubAgentInput(BaseModel):
        """Defines the input argument for our agent tool."""
        input: str = Field(description="A detailed, natural language query for the agent.")
        context: Optional[list[list]] = Field(default=[], description="Optional context")

    # Define the tool function
    async def run_agent(input: str, context: Optional[list[str]] = []) -> str:
        """The async function that will be executed when the tool is called."""

        agent_input = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=input)
            ]
        }

        if context:
            agent_input["context"] = context

        print(f"\n\nFunction Call:\n\tinput={input}, context={context}")
        result = await agent_executor.ainvoke(agent_input)
        print(f"Function Call: result: {result}")
        return result['messages'][-1].content

    # The parameter is changed from 'func' to 'coroutine' to correctly
    # handle the async function. This is the only change needed.
    return StructuredTool.from_function(
        name=tool_name,
        description=tool_description,
        coroutine=run_agent,  # <-- THE FIX IS HERE
        args_schema=SubAgentInput
    )

async def create_agent(config: dict, llm=None) -> BaseTool:
    system_prompt = (
        "You are an agent equipped with a set of MCP tools. Use these tools to accurately fulfill user requests.\n\n"
        "Each tool has a specific function signature, input requirements, and output format. Read them carefully before selecting and invoking a tool.\n\n"
        "- Always choose the most relevant tool based on the task.\n"
        "- Strictly follow the input type and parameter names as described.\n"
        "- If `context` is provided, use it to improve the accuracy of your answer.\n"
        "- Do not fabricate tool outputs. Only return what the tool provides.\n"
        "- You can optionally provide a list of strings as `context` to clarify any ambiguity (e.g., time of day, elevation, past weather).\n"
        "- This tool does NOT retain the output of previous calls. If chaining values (e.g., using temperature in math), you MUST explicitly pass prior outputs via `context`.\n"
    )

    # 1. Create the specialized agent executor
    mcp_agent_executor = await create_mcp_react_agent(config, llm)
    print("\n--- Agent Executor Created ---")

    # 2. Wrap the agent executor in a StructuredTool
    name = config["name"]
    description = config["description"]
    agent_tool = create_agent_as_tool(
        agent_executor=mcp_agent_executor,
        system_prompt=system_prompt,
        tool_name=name,
        tool_description=description
    )
    print("\n--- Agent Wrapped as a Tool ---")
    print(f"Tool Name: {agent_tool.name}")
    print(f"Tool Description: {agent_tool.description}")
    print(f"Tool Args: {list(agent_tool.args.keys())}")

    return agent_tool

# def get_agent_client(config: dict, llm=None) -> BaseTool:
#     name = config["name"]
#     mcp_config = config["mcp"]



async def create_weather_agent(llm):
    agent_config = {
        "name": "weather_agent",
        "description": "Useful for answering questions about the current weather, temperature, fine dust levels, and precipitation chance in a specific city.",
        "mcp": {
            "transport": "streamable_http",
            "url": "http://localhost:8001/mcp",
        }
    }

    return await create_agent(agent_config, llm)

async def create_knox_mail_agent_tool(llm):
    agent_config = {
        "name": "knox_mail_agent",
        "description": "Useful for sending mail, listing unread mail, and read mail",
        "mcp": {
            "transport": "streamable_http",
            "url": "http://localhost:8002/mcp",
        }
    }

    return await create_agent(agent_config, llm)





async def main():
    """
    Main function to configure the agent, wrap it as a tool, and invoke it.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)


    weather_agent_tool = await create_weather_agent(llm)

    # 3. Demonstrate invoking the new tool
    print("\n--- Invoking the Tool ---")
    query = "What is the fine dust level in Busan?"
    
    # This ainvoke call was already correct. It will now work as expected.
    tool_result = await weather_agent_tool.ainvoke({"input": query})
    
    print(f"\nQuery: '{query}'")
    print(f"Result from tool: {tool_result}")

    query2 = "What is the weather like in Busan and what is the fine dust level there?"
    print("#############################################")
    print(f"Query 2: {query2}")
    tool_result2 = await weather_agent_tool.ainvoke({"input": query2})

    print(f"Result from tool: {tool_result2}")
    
    print("\n#############################################\n")
    print("\nKnox Mail Test")
    knox_mail_agent_tool = await create_knox_mail_agent_tool(llm)

    print("\n--- Sending mail to a person ---")
    query = "Email at ionia.kim@samsung.com with the subject 'Hi,' the message I'm fine."
    
    # This ainvoke call was already correct. It will now work as expected.
    tool_result = await knox_mail_agent_tool.ainvoke({"input": query})
    
    print(f"\nQuery: '{query}'")
    print(f"Result from tool: {tool_result}")

    print("\n--- Composite Task")
    query = "Read the first mail from unread mail"
    tool_result = await knox_mail_agent_tool.ainvoke({"input": query})

    print(f"\nQuery: '{query}'")
    print(f"Result from tool: {tool_result}")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure your MCP server is running at http://localhost:8001/mcp and that the tools are correctly defined.")