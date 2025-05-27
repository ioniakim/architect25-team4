import gradio as gr
import requests


with gr.Blocks() as demo:
    gr.Markdown('# [DEMO] Multi-Agents Orchestration System 🤖')

    with gr.Row():
        chatbot = gr.Chatbot(show_label=False)
    with gr.Row():
        input_textbox = gr.Textbox(show_label=False, placeholder='요청을 입력하세요.', lines=2)
    send_button = gr.Button('SEND')

    def _chat(user_message, history):
        history = history or []
        history.append([user_message, ''])
        return '', history

    def stream_result(history):
        url = 'http://localhost:8000/test'
        data = {"message": history[-1][0]}
        with requests.post(url, json=data, stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    history[-1][1] = chunk.decode('utf-8')
                    history.append([None, ''])
                    yield history
        if history[-1][0] is None:
            history = history[:-1]
        yield history

    send_button.click(
        _chat, inputs=[input_textbox, chatbot], outputs=[input_textbox, chatbot]
    ).then(
        stream_result, inputs=chatbot, outputs=chatbot
    )


if __name__ == "__main__":
    demo.launch()
