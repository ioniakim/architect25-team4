try:
    from ..managers.llm_manager import LLM
    from ..managers.tool_manager import ToolManager
except ImportError:
    from managers.llm_manager import LLM
    from managers.tool_manager import ToolManager
try:
    from .tools.math_tool import get_math_tool
except ImportError:
    from tools.math_tool import get_math_tool

#from .tools.weather_tool import get_weather_tool
from .tools.mcp_warpper_tool import get_weather_agent_tool
from .tools.mcp_warpper_tool_mail import get_mail_agent_tool
from .tools.mcp_warpper_tool_jira import get_jira_agent_tool
from .tools.mcp_warpper_tool_calendar import get_calendar_agent_tool
from .tools.mcp_warpper_tool_contact import get_contact_agent_tool


def prepare():
    #LLM.set(model='', api_key='empty', base_url='http://34.64.195.131:80/v1')
    
    LLM.set(model='gpt-4.1-mini', api_key=API_KEY, base_url='https://api.openai.com/v1', temperature=0)
    _calculate = get_math_tool(LLM.get())

    ToolManager.set(_calculate)
    #ToolManager.set(get_weather_tool(), name='get_weather_info')
    ToolManager.set(get_weather_agent_tool(), name='weather_agent')
    ToolManager.set(get_calendar_agent_tool(), name='calendar_agent')
    ToolManager.set(get_mail_agent_tool(), name='mail_agent')
    ToolManager.set(get_jira_agent_tool(), name='jira_agent')
    ToolManager.set(get_contact_agent_tool(), name='contact_agent')
