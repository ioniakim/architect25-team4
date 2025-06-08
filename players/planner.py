################################################################################
# Planner
################################################################################

from typing import Sequence
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import FunctionMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch
from langchain_core.tools import BaseTool
from .output_parser import LLMCompilerPlanParser


def build(
        model: BaseChatModel,
        tools: Sequence[BaseTool],
        prompt_template: ChatPromptTemplate,
        replanner_description: str,
) -> Runnable:
    num_tools = len(tools) + 1  # Add one because we're adding the join() tool at the end.
    tool_descriptions = '\n'.join(f'{n}. {tool.description}\n' for n, tool in enumerate(tools, 1))
    planner_prompt = prompt_template.partial(
        replan='', num_tools=num_tools, tool_descriptions=tool_descriptions)
    replanner_prompt = prompt_template.partial(
        replan=replanner_description, num_tools=num_tools, tool_descriptions=tool_descriptions)

    def should_replan(state: list):
        # print(f'@@ {__file__} >> should_replan: {state} -> {isinstance(state[-1], SystemMessage)}')
        # Context is passed as a system message
        return isinstance(state[-1], SystemMessage)

    def wrap_messages(state: list):
        # print(f'@@ {__file__} >> wrap_messages: {state}')
        return {"messages": state}

    def wrap_and_get_last_index(state: list):
        # print(f'@@ {__file__} >> wrap_and_get_last_index: input={state}')
        next_task = 0
        for message in state[::-1]:
            if isinstance(message, FunctionMessage):
                next_task = message.additional_kwargs["idx"] + 1
                break
        state[-1].content = state[-1].content + f" - Begin counting at : {next_task}"
        # print(f'@@ {__file__} >> wrap_and_get_last_index: output={state}')
        return {"messages": state}

    return (
        RunnableBranch(
            (should_replan, wrap_and_get_last_index | replanner_prompt),
            wrap_messages | planner_prompt,
        )
        | model
        | LLMCompilerPlanParser(tools=tools)
    )
