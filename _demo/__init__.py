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

from .tools.weather_tool import get_weather_tool
from .tools.mcp_warpper_tool import get_weather_agent_tool


def prepare():
    #LLM.set(model='', api_key='empty', base_url='http://34.64.195.131:80/v1')
    
    
    LLM.set(model='gpt-4.1-mini', api_key=API_KEY, base_url='https://api.openai.com/v1')
    _calculate = get_math_tool(LLM.get())

    ToolManager.set(_calculate)
    #ToolManager.set(get_weather_tool(), name='get_weather_info')
    ToolManager.set(get_weather_agent_tool(), name='get_weather_agent')
