import sys
import os
import io
import tempfile
import re
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from graph.workflow import create_workflow
from tts_asr.asr_whisper import transcribe
from tts_asr.tts_client import synthesize

# Initialize the agent graph
agent_graph = create_workflow()

def process_query(text_input, audio_input, input_method):
    """Process the user query and return results"""
    
    if input_method == "Text" and not text_input:
        return "Please enter a search query", "", "", "", "", ""
    
    if input_method == "Voice" and not audio_input:
        return "Please record a voice message", "", "", "", "", ""
    
    # Determine the final query
    final_query = text_input
    transcription_result = ""
    
    if input_method == "Voice" and audio_input:
        final_query = transcribe(audio_input, os.getenv("ASR_MODEL", "small"))
        transcription_result = f"Transcribed: {final_query}"
    
    # Process with agent
    initial_state = {
        "audio_path": None,
        "transcript": final_query,
        "intent": None,
        "plan": None,
        "evidence": None,
        "answer": None,
        "citations": None,
        "safety_flags": None,
        "tts_path": None,
        "log": []
    }
    
    result = agent_graph.invoke(initial_state)
    
    # Extract data
    answer = result.get("answer", "No response generated")
    evidence = result.get("evidence") or {}
    products = evidence.get("rag", [])
    web_results = evidence.get("web", [])
    citations = result.get("citations", [])
    logs = result.get("log", [])
    
    # Format products
    products_html = ""
    if products:
        products_html = "<div style='margin-top: 20px;'>"
        for idx, product in enumerate(products[:5], 1):
            brand = product.get("brand", "")
            price = product.get("price", "")
            rating = product.get("rating", "")
            image_urls = product.get("image_urls", [])
            
            tags = []
            if brand:
                tags.append(f'<span style="background: #667eea; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; margin-right: 8px;">{brand}</span>')
            if price:
                tags.append(f'<span style="background: #10b981; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; margin-right: 8px;">${price}</span>')
            if rating:
                tags.append(f'<span style="background: #f59e0b; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; margin-right: 8px;">{rating} stars</span>')
            
            product_html = f"""
            <div style="background: #2a2a2a; padding: 20px; margin-bottom: 15px; border-left: 4px solid #667eea;">
                <h4 style="color: #ffffff; margin-bottom: 10px;">{idx}. {product.get("title", "Unknown Product")}</h4>
                <div>{''.join(tags)}</div>
            """
            
            if image_urls:
                product_html += "<div style='display: flex; gap: 10px; margin-top: 10px;'>"
                for image_url in image_urls:
                    product_html += f"<img src='{image_url}' style='width: 100px; height: 100px; object-fit: cover; border-radius: 8px;'/>"
                product_html += "</div>"
            
            product_html += "</div>"
            products_html += product_html
        products_html += "</div>"
    else:
        products_html = "<p style='color: #cccccc;'>No products found in catalog</p>"
    
    # Format web results
    web_html = ""
    if web_results:
        web_html = "<div style='margin-top: 20px;'>"
        for item in web_results[:5]:
            web_html += f"""
            <div style="background: #f8fafc; border-left: 4px solid #667eea; padding: 15px; margin-bottom: 10px; border-radius: 8px;">
                <a href="{item.get('url', '#')}" target="_blank" style="color: #667eea; font-weight: 600; text-decoration: none;">{item.get('title', 'Link')}</a>
                <p style="color: #64748b; margin-top: 8px; font-size: 14px;">{item.get('snippet', '')[:200]}</p>
            </div>
            """
        web_html += "</div>"
    else:
        web_html = "<p>No web results available</p>"
    
    # Format citations
    citations_html = ""
    if citations:
        citations_html = "<div style='margin-top: 20px;'>"
        for cite in citations:
            if cite.get("doc_id"):
                citations_html += f'<p style="color: #64748b; margin-bottom: 8px;">Document: {cite.get("doc_id")}</p>'
            elif cite.get("url"):
                citations_html += f'<p style="color: #64748b; margin-bottom: 8px;">{cite.get("url")}</p>'
        citations_html += "</div>"
    else:
        citations_html = "<p>No citations available</p>"
    
    # Format logs
    logs_text = ""
    if logs:
        for idx, log_entry in enumerate(logs, 1):
            logs_text += f"Step {idx}: {log_entry.get('node', 'Unknown').upper()}\n"
            logs_text += f"{str(log_entry)}\n\n"
    else:
        logs_text = "No logs available"
    
    # Metrics
    metrics_html = f"""
    <div style="display: flex; gap: 30px; margin: 20px 0; flex-wrap: wrap; background: #1a1a1a; padding: 20px;">
        <div style="text-align: center; flex: 1; min-width: 120px;">
            <div style="font-size: 2rem; font-weight: 700; color: #ffffff;">{len(products)}</div>
            <div style="color: #cccccc; font-size: 14px;">Products Found</div>
        </div>
        <div style="text-align: center; flex: 1; min-width: 120px;">
            <div style="font-size: 2rem; font-weight: 700; color: #ffffff;">{len(web_results)}</div>
            <div style="color: #cccccc; font-size: 14px;">Web References</div>
        </div>
        <div style="text-align: center; flex: 1; min-width: 120px;">
            <div style="font-size: 2rem; font-weight: 700; color: #ffffff;">{len(citations)}</div>
            <div style="color: #cccccc; font-size: 14px;">Citations</div>
        </div>
    </div>
    """
    
    return transcription_result, answer, metrics_html, products_html, web_html, citations_html, logs_text

def generate_audio(answer):
    """Generate audio from text answer"""
    if not answer or answer == "No response generated":
        return None
    
    # Remove citations from answer
    clean_answer = re.sub(r'\(Sources?:.*?\)', '', answer).strip()
    
    try:
        audio_path = synthesize(clean_answer)
        return audio_path
    except Exception as e:
        print(f"Audio generation failed: {e}")
        return None

# Create Gradio interface
with gr.Blocks(title="AI Product Assistant") as app:
    
    # Header
    gr.HTML("""
    <div style="text-align: center; padding: 40px 20px; background: #1a1a1a; margin-bottom: 30px;">
        <h1 style="font-size: 2.5rem; font-weight: 700; color: #ffffff; margin: 0;">AI Product Assistant</h1>
        <p style="font-size: 1.1rem; color: #cccccc; margin-top: 10px;">Your intelligent shopping companion powered by advanced AI agents</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column():
            # Input method selection
            input_method = gr.Radio(
                choices=["Text", "Voice"],
                value="Text",
                label="Choose input method",
                interactive=True
            )
            
            # Text input
            text_input = gr.Textbox(
                label="Enter your product search query",
                placeholder="Example: Find an eco-friendly dish soap under $12 with natural ingredients",
                lines=2,
                visible=True
            )
            
            # Audio input
            audio_input = gr.Audio(
                label="Record your voice query",
                type="filepath",
                visible=False
            )
            
            # Search button
            search_btn = gr.Button("Search Products", variant="primary", size="lg")
    
    # Results section
    with gr.Row():
        with gr.Column():
            # Transcription result
            transcription_output = gr.Textbox(
                label="Voice Transcription",
                visible=False,
                interactive=False
            )
            
            # AI Response
            answer_output = gr.Textbox(
                label="AI Response",
                lines=5,
                interactive=False
            )
            
            # Metrics
            metrics_output = gr.HTML(label="Metrics")
    
    # Tabs for detailed results
    with gr.Tabs():
        with gr.Tab("Products"):
            products_output = gr.HTML()
        
        with gr.Tab("Web Results"):
            web_output = gr.HTML()
        
        with gr.Tab("Citations"):
            citations_output = gr.HTML()
        
        with gr.Tab("Agent Log"):
            logs_output = gr.Textbox(
                lines=10,
                interactive=False
            )
        
        with gr.Tab("Audio"):
            with gr.Row():
                audio_btn = gr.Button("Generate Audio Response")
                audio_output = gr.Audio(label="Audio Response")
    
    # Event handlers
    def toggle_inputs(method):
        if method == "Text":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)
    
    def show_transcription(method):
        return gr.update(visible=(method == "Voice"))
    
    input_method.change(
        toggle_inputs,
        inputs=[input_method],
        outputs=[text_input, audio_input]
    )
    
    input_method.change(
        show_transcription,
        inputs=[input_method],
        outputs=[transcription_output]
    )
    
    # Search functionality
    search_btn.click(
        process_query,
        inputs=[text_input, audio_input, input_method],
        outputs=[
            transcription_output,
            answer_output,
            metrics_output,
            products_output,
            web_output,
            citations_output,
            logs_output
        ]
    )
    
    # Audio generation
    audio_btn.click(
        generate_audio,
        inputs=[answer_output],
        outputs=[audio_output]
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )