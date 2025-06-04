import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

from langchain_core.messages import HumanMessage
from managers.llm_manager import LLM
from managers.tool_manager import ToolManager
from managers.prompt_manager import PromptManager
from conductor import build
from _demo import prepare


prepare()
conductor = build(LLM.get(), ToolManager.list(), PromptManager.get(LLM.name()))


async def generate_response(user_message: str) -> AsyncGenerator[bytes, None]:
    print('\n########## START ##########\n')
    yield '[Processing]'
    await asyncio.sleep(0.5)
    for step in conductor.stream({"messages": [HumanMessage(content=user_message)]}):
        print('========== <STEP> ==========')
        print(step)
        print('============================')
        yield str(step).encode('utf-8')
        await asyncio.sleep(0.5)
    yield '[Done]'
    print('\n########## DONE ##########\n')


app = FastAPI()


@app.post('/test')
async def test(request: Request):
    data = await request.json()
    return StreamingResponse(generate_response(data.get("message", '')), media_type='text/plain')


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=3000)
