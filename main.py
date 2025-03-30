import logging
import time
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from modelscope import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from qwen_infer import qwen_chat, qwen_stream_chat


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-1.8B-Chat",
        torch_dtype="auto",
        device_map="auto"
    )
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")

homepage_html_for_websocket = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
        <style>
            #messages {
                list-style: none;
                padding: 0;
                margin: 0;
                display: flex;       /* 启用弹性布局 */
                flex-wrap: wrap;     /* 允许换行 */
                gap: 10px;          /* 消息间隔 */
                max-width: 100vw;   /* 限制最大宽度 */
            }
            #messages li {
                background: #f0f0f0;
                padding: 5px 10px;
                border-radius: 4px;
                white-space: pre-wrap; /* 保持单行不换行 */
            }
        </style>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://localhost:8000/chat");
            var currentMessage = null;
        
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages');
                var data = event.data;
        
                // 场景 1：持续流式追加
                if (!currentMessage) {
                    currentMessage = document.createElement('li');
                    messages.appendChild(currentMessage);
                }
                currentMessage.textContent += data;
            };
        
            function sendMessage(event) {
                var input = document.getElementById("messageText");
                ws.send(input.value);
                input.value = '';
                event.preventDefault();
            }
        </script>
    </body>
</html>
"""


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


@app.get("/")
async def get():
    return HTMLResponse(homepage_html_for_websocket)


@app.post("/chat", response_model=ChatCompletionResponse)
async def chat(request: ChatCompletionRequest):
    global model, tokenizer

    print("post request:", request)
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content

    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        query = prev_messages.pop(0).content + query

    print("post query before chat:", query)
    print("post history before chat:", prev_messages)
    response, history = qwen_chat(model, tokenizer, query, history=prev_messages)
    print("post response after chat:", response)
    print("post history after chat:", history)
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )

    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")


@app.websocket("/chat")
async def chat(websocket: WebSocket):
    """
    input: JSON String of {"query": "", "history": []}
    output: JSON String of {"response": "", "history": [], "status": 200}
        status 200 stand for response ended, else not
    """
    await websocket.accept()
    messages = []
    try:
        while True:
            query = await websocket.receive_text()
            print("Web socket request before stream_chat:", query)
            if query == 'quit!':
                print("Web Socket识别到quit!断开。")
                break
            messages.append({"role": "user", "content": query})
            prev_messages = messages[:-1]
            if len(prev_messages) > 0 and prev_messages[0]['role'] == "system":
                query = prev_messages.pop(0)['content'] + query
            print("Web socket prev_messages before stream_chat:", prev_messages)
            print("Web socket messages before stream_chat:", messages)
            new_content = ""
            for response in qwen_stream_chat(model, tokenizer, query, history=prev_messages):
                res = {
                    "response": response,
                    "history": prev_messages,
                    "status": 202,
                }
                new_content += response
                await websocket.send_text(response)
                print("Web socket response:", res)
            messages += [{'role': 'user', 'content': query}]
            messages += [{'role': 'assistant', 'content': new_content}]

            print("Web socket history final:", messages)
            await websocket.send_json({"status": 200})
    except WebSocketDisconnect:
        print("Web Socket异常断开！")
    finally:
        await websocket.close()


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-1.8B-Chat",
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")

    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
