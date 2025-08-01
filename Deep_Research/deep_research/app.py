from pathlib import Path
from dotenv import load_dotenv

# بارگذاری .env از ریشهٔ پروژه
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

import asyncio
import gradio as gr
from .research_manager import ResearchManager   # ← import نسبی

async def run_research(query: str):
    if not query.strip():
        yield "❗️ Please enter a research query."
        return
    manager = ResearchManager()
    async for chunk in manager.run(query):
        yield chunk

with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as ui:
    gr.Markdown("# 🔬 Deep Research")
    gr.Markdown("Enter a topic to research and get a comprehensive report.")
    query = gr.Textbox(label="Topic", placeholder="e.g. Quantum computing", lines=2)
    run_btn = gr.Button("🚀 Start Research", variant="primary")
    report = gr.Markdown("*Your research report will appear here…*")
    run_btn.click(run_research, inputs=query, outputs=report)
    query.submit(run_research, inputs=query, outputs=report)

if __name__ == "__main__":
    ui.launch(inbrowser=True, share=False)
