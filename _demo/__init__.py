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


def prepare():
    LLM.set(model='', api_key='empty', base_url='http://34.64.195.131:80/v1')
    _calculate = get_math_tool(LLM.get())
    ToolManager.set(_calculate)
