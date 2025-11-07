# Simple Gradio UI to interact with the API and agent locally
import gradio as gr
import requests
import os

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
#create session
session = requests.post(f"{API_BASE}/apps/{"Agent"}/users/{"mani001"}/sessions")
SESSION_ID = session.json()['id']
print(f"Session Creation Response: {SESSION_ID}")

def call_agent(prompt):
    r = requests.post(f"{API_BASE}/run", json={
                        "app_name": "Agent",
                        "user_id": "mani001",
                        "session_id": SESSION_ID,
                        "new_message": {
                            "role": "user",
                            "parts": [
                            {
                                "text": prompt
                            }
                            ]
                        },
                        "streaming": "false"
                        })
    print(f"API Call Status Code: {r.json()[2]}")
    print(f"API Call Response Text: {r.json()}")
    return r.json()[2]['actions']['state_delta']['response']

def call_qa(q):
    r = requests.post(f"{API_BASE}/tools/qa", json={"prompt": q})
    return r.json()['answer']

def call_summarize(q):
    r = requests.post(f"{API_BASE}/tools/summarize", json={"prompt": q})
    print(f"API Call Status Code: {r.status_code}")
    print(f"API Call Response Text: {r.text}")
    return r.json()['summary']

def call_extract(q):
    r = requests.post(f"{API_BASE}/tools/extract", json={"prompt": q})
    return r.json()['extraction']

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
