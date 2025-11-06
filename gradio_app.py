# Gradio.py
# Simple Gradio UI to interact with the API and agent locally
import gradio as gr
import requests
import os

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

def call_agent(query):
    r = requests.post(f"{API_BASE}/agent", json={"query": query})
    return r.json()

def call_qa(q):
    r = requests.post(f"{API_BASE}/qa", json={"question": q})
    return r.json()

def call_summarize(q):
    r = requests.post(f"{API_BASE}/summarize", json={"prompt": q})
    print(f"API Call Status Code: {r.status_code}")
    print(f"API Call Response Text: {r.text}")
    return r.json()

def call_extract(q):
    r = requests.post(f"{API_BASE}/extract", json={"field_prompt": q})
    return r.json()

with gr.Blocks(title="AI Market Analyst") as demo:
    gr.Markdown("# AI Market Analyst\nInteract with the tools or the conversational agent.")
    with gr.Row():
        with gr.Column():
            tool_input = gr.Textbox(label="Tool Input / Prompt")
            qa_btn = gr.Button("QA")
            sum_btn = gr.Button("Summarize")
            ex_btn = gr.Button("Extract")
            tool_output = gr.Textbox(label="Tool Output", lines=12)
        with gr.Column():
            conv_in = gr.Textbox(label="Conversational Query")
            conv_btn = gr.Button("Run Agent")
            conv_out = gr.Textbox(label="Agent Output", lines=12)

    def do_qa(x):
        return str(call_qa(x))
    def do_sum(x):
        return str(call_summarize(x))
    def do_ex(x):
        return str(call_extract(x))
    def do_agent(x):
        return str(call_agent(x))

    qa_btn.click(do_qa, inputs=[tool_input], outputs=[tool_output])
    sum_btn.click(do_sum, inputs=[tool_input], outputs=[tool_output])
    ex_btn.click(do_ex, inputs=[tool_input], outputs=[tool_output])
    conv_btn.click(do_agent, inputs=[conv_in], outputs=[conv_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
