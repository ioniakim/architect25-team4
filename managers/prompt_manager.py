################################################################################
# Prompts
################################################################################

from langchain import hub
from langchain_core.prompts import ChatPromptTemplate



# TODO:
plan_prompt_template_path = '/home/user/bogun/source/architect25-team4/managers/plan.json'
join_prompt_template_path = '/home/user/bogun/source/architect25-team4/managers/join.json'
# hub.

_plan: ChatPromptTemplate = hub.pull("wfh/llm-compiler")
_join: ChatPromptTemplate = hub.pull("wfh/llm-compiler-joiner")
_replan: str = \
    ' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results' \
    ' (given as Observation) of each plan and a general thought (given as Thought) about the executed results.' \
    ' You MUST use these information to create the next plan under "Current Plan".\n' \
    ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n' \
    ' - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.\n' \
    ' - You must continue the task index from the end of the previous one. Do not repeat task indices.'


# TODO: simple memory DB
_default_key = "default"
_DATA: dict[str, dict[str, ChatPromptTemplate | str]] = {
    "default": {"plan": _plan, "replan": _replan, "join": _join},
}


class PromptManager:
    @staticmethod
    def get(key: str) -> ChatPromptTemplate | str | None:
        global _DATA
        return _DATA.get(key, _DATA[_default_key])
