import os
try:
    from ..managers.llm_manager import LLM
    from ..managers.tool_manager import ToolManager
except ImportError:
    from managers.llm_manager import LLM
    from managers.tool_manager import ToolManager
try:
    from .tools.math_tool import get_math_tool
    from .tools.search_tool import get_search_tool
except ImportError:
    from tools.math_tool import get_math_tool
    from tools.search_tool import get_search_tool
    from tools.weather_tool import get_weather_tool

#from .tools.weather_tool import get_weather_tool
from .tools.mcp_warpper_tool import get_weather_agent_tool
from .tools.mcp_warpper_tool_mail import get_mail_agent_tool
from .tools.mcp_warpper_tool_jira import get_jira_agent_tool
from .tools.mcp_warpper_tool_calendar import get_calendar_agent_tool
from .tools.mcp_warpper_tool_contact import get_contact_agent_tool


def prepare():
    openai_api_key = os.getenv("OPENAI_API_KEY", None)
    gemini_api_key = os.getenv("GEMINI_API_KEY", None)
    if openai_api_key is not None:
        model = os.getenv("OPENAI_MODEL", 'gpt-4.1-nano')
        api_key = openai_api_key
        base_url = 'https://api.openai.com/v1'
        max_tokens = 1024
        print(f'OpenAI: {model}')
    elif gemini_api_key is not None:
        model = os.getenv("GEMINI_MODEL", None)
        api_key = gemini_api_key
        base_url = None
        max_tokens = 1024
        print(f'Gemini: {model}')
    else:
        model = ''
        api_key = 'empty'
        base_url = 'http://34.64.195.131:80/v1'
        max_tokens = 1024
        print(f'LLM: {base_url}')
    LLM.set(model=model, api_key=api_key, base_url=base_url, max_tokens=max_tokens, temperature=0)

    ToolManager.set(get_math_tool(LLM.get()))
    #ToolManager.set(get_weather_tool(), name='get_weather_info')
    ToolManager.set(get_weather_agent_tool(), name='weather_agent')
    ToolManager.set(get_calendar_agent_tool(), name='calendar_agent')
    ToolManager.set(get_mail_agent_tool(), name='mail_agent')
    ToolManager.set(get_jira_agent_tool(), name='jira_agent')
    ToolManager.set(get_contact_agent_tool(), name='contact_agent')
