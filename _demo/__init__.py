import os
try:
    from ..llm import get_llm_client
    from ..managers.llm_manager import LLM
    from ..managers.tool_manager import ToolManager
except ImportError:
    from llm import get_llm_client
    from managers.llm_manager import LLM
    from managers.tool_manager import ToolManager
try:
    from .tools.math_tool import get_math_tool
    from .tools.search_tool import get_search_tool
    from .tools.test_tool import get_test_tools
except ImportError:
    from tools.math_tool import get_math_tool
    from tools.search_tool import get_search_tool
    from tools.weather_tool import get_weather_tool
    from tools.test_tool import get_test_tools

#from .tools.weather_tool import get_weather_tool
from .tools.mcp_warpper_tool import get_weather_agent_tool
from .tools.mcp_warpper_tool_mail import get_mail_agent_tool
from .tools.mcp_warpper_tool_jira import get_jira_agent_tool
from .tools.mcp_warpper_tool_calendar import get_calendar_agent_tool
from .tools.mcp_warpper_tool_contact import get_contact_agent_tool


def prepare():
    openai_api_key = os.getenv("OPENAI_API_KEY", None)
    google_api_key = os.getenv("GOOGLE_API_KEY", None)
    if openai_api_key is not None:
        llm_type = "OPENAI"
        config = {
            "api_key": openai_api_key,
            "model": os.getenv("OPENAI_MODEL", 'gpt-4.1-nano'),
            "base_url": 'https://api.openai.com/v1',
            "max_tokens": 1024,
            "temperature": 0,
        }
        print(f'OpenAI: {config["model"]}')
    elif google_api_key is not None:
        llm_type = "GOOGLE"
        config = {
            "api_key": google_api_key,
            "model": os.getenv("GOOGLE_MODEL", 'gemini-2.0-flash-lite'),
            "base_url": None,
            "max_tokens": 1024,
            "temperature": 0,
        }
        print(f'Google: {config["model"]}')
    else:
        llm_type = "OPENAI"
        config = {
            "api_key": "EMPTY",
            "model": '',
            "base_url": 'http://34.64.195.131:80/v1',
            "max_tokens": 1024,
            "temperature": 0,
        }
        print(f'LLM: {config["base_url"]}')
    LLM.set(get_llm_client(llm_type, config))

    # --- Add the test tools ---
    for tool in get_test_tools():
        ToolManager.set(tool)
    # --------------------------

    ToolManager.set(get_math_tool(LLM.get()))
    # ToolManager.set(get_weather_agent_tool(), name='weather_agent')
    # ToolManager.set(get_calendar_agent_tool(), name='calendar_agent')
    # ToolManager.set(get_mail_agent_tool(), name='mail_agent')
    # ToolManager.set(get_jira_agent_tool(), name='jira_agent')
    # ToolManager.set(get_contact_agent_tool(), name='contact_agent')

