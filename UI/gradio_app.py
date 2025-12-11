import sys
import os
import re
import json
import logging
import httpx
from bs4 import BeautifulSoup
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv

# --- Setup Paths & Imports ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from graph.workflow import create_workflow
from tts_asr.asr_whisper import transcribe
from tts_asr.tts_client import synthesize
from graph.llm_interface import get_llm_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

agent_graph = create_workflow()
llm_client = get_llm_client()

# --- Helpers ---

def enrich_web_result(url):
    """Scrapes Amazon/Web pages for high-quality product images and metadata."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    data = {}
    try:
        with httpx.Client(timeout=3, headers=headers, follow_redirects=True) as client:
            response = client.get(url)
            if response.status_code != 200: return {}
            soup = BeautifulSoup(response.text, "html.parser")
            
            # --- 1. Image Extraction (Prioritize Product Images over Logos) ---
            # Try Amazon main image ID first
            main_img = soup.select_one("#landingImage, #imgBlkFront")
            if main_img:
                # Amazon often puts the high-res URL in 'src' or 'data-old-hires'
                data["image_url"] = main_img.get("src") or main_img.get("data-old-hires")
            
            # If failed, look for dynamic image container (common in modern Amazon layouts)
            if not data.get("image_url"):
                dynamic_img = soup.select_one("#landingImage")
                if dynamic_img and dynamic_img.get("data-a-dynamic-image"):
                    # This attribute contains a JSON string of image URLs
                    try:
                        img_dict = json.loads(dynamic_img.get("data-a-dynamic-image"))
                        # Get the largest image key
                        if img_dict:
                            data["image_url"] = list(img_dict.keys())[0]
                    except:
                        pass

            # Fallback: Open Graph Image (but filter out common logo filenames)
            if not data.get("image_url"):
                og_img = soup.select_one("meta[property='og:image']")
                if og_img:
                    url = og_img.get("content", "")
                    # Simple heuristic to avoid generic amazon logos
                    if "amazon_logo" not in url and "nav-logo" not in url:
                        data["image_url"] = url

            # --- 2. Rating & Brand Extraction ---
            rating_tag = soup.select_one("span.a-icon-alt")
            if rating_tag:
                data["rating"] = rating_tag.get_text(strip=True).split(" ")[0]
            
            brand_tag = soup.select_one("#bylineInfo")
            if brand_tag:
                data["brand"] = brand_tag.get_text(strip=True).replace("Visit the ", "").replace(" Store", "")
            
            return data
    except:
        return {}

def llm_generate_comparison(products, web_results, user_query):
    """Generates a clean HTML comparison table using the LLM."""
    if not products and not web_results:
        return "<p>No data to compare.</p>"

    # Prepare Context
    items_context = []
    for p in products[:4]:
        items_context.append({
            "source": "Catalog",
            "title": p.get("title"),
            "price": p.get("price"),
            "brand": p.get("brand"),
            "rating": p.get("rating"),
            "snippet": p.get("snippet", "")
        })
    for w in web_results[:4]:
        items_context.append({
            "source": "Web",
            "title": w.get("title"),
            "price": w.get("price"),
            "brand": w.get("brand"),
            "rating": w.get("rating"),
            "snippet": w.get("snippet", "")
        })

    prompt = f"""
    You are a shopping assistant. Compare these products for the query: "{user_query}".
    Raw Data: {json.dumps(items_context)}

    Task:
    1. Infer 'Brand' and 'Rating' from titles/snippets if missing.
    2. Write a short 'Verdict' on the best option.
    3. Return JSON:
    {{
        "verdict": "Recommendation text...",
        "rows": [
            {{ "name": "Title", "price": "$X", "brand": "Y", "rating": "4.5", "source": "Web/Catalog", "pros": "Key feature" }}
        ]
    }}
    """
    
    try:
        response_json = llm_client.generate_json([{"role": "user", "content": prompt}])
        verdict = response_json.get("verdict", "See comparison below.")
        rows = response_json.get("rows", [])
        
        html = f"""
        <div style="background: #27272a; padding: 15px; border-radius: 6px; margin-bottom: 20px; border-left: 4px solid #10b981;">
            <strong style="color: #fff; display:block; margin-bottom:5px;">AI Verdict</strong>
            <span style="color: #d4d4d8;">{verdict}</span>
        </div>
        <div style="overflow-x: auto;">
            <table style="width: 100%; border-collapse: collapse; color: #e4e4e7; font-size: 0.9em;">
                <thead>
                    <tr style="background: #18181b; border-bottom: 2px solid #3f3f46;">
                        <th style="padding: 10px; text-align: left;">Product</th>
                        <th style="padding: 10px; text-align: left;">Price</th>
                        <th style="padding: 10px; text-align: left;">Brand</th>
                        <th style="padding: 10px; text-align: left;">Rating</th>
                        <th style="padding: 10px; text-align: left;">Source</th>
                    </tr>
                </thead>
                <tbody>
        """
        for i, row in enumerate(rows):
            bg = "#27272a" if i % 2 == 0 else "#18181b"
            badge_col = "#059669" if "Catalog" in row['source'] else "#4f46e5"
            html += f"""
            <tr style="background: {bg}; border-bottom: 1px solid #3f3f46;">
                <td style="padding: 10px;">
                    <div style="font-weight: 600;">{row['name']}</div>
                    <div style="font-size: 0.85em; color: #a1a1aa;">{row.get('pros','')}</div>
                </td>
                <td style="padding: 10px; color: #34d399; font-weight: bold;">{row['price']}</td>
                <td style="padding: 10px;">{row['brand']}</td>
                <td style="padding: 10px; color: #fbbf24;">{row['rating']}</td>
                <td style="padding: 10px;"><span style="background:{badge_col}; color:#fff; padding:2px 8px; border-radius:10px; font-size:0.75em;">{row['source']}</span></td>
            </tr>
            """
        html += "</tbody></table></div>"
        return html
    except:
        return "<p>Comparison generation failed.</p>"

def process_query(text_input, audio_input, input_method):
    # 1. Inputs
    if input_method == "Text" and not text_input: 
        return gr.update(visible=False), "Enter query", "", "", "", "", "", ""
    if input_method == "Voice" and not audio_input: 
        return gr.update(visible=False), "Record audio", "", "", "", "", "", ""

    final_query = text_input
    transcription_update = gr.update(visible=False)
    
    if input_method == "Voice" and audio_input:
        final_query = transcribe(audio_input, os.getenv("ASR_MODEL", "small"))
        # Make transcription box visible with the text
        transcription_update = gr.update(visible=True, value=f"🎤 Transcribed: {final_query}")

    # 2. Agent Workflow
    initial_state = {"transcript": final_query, "log": []}
    result = agent_graph.invoke(initial_state)
    
    answer = result.get("answer", "No response.")
    products = result.get("evidence", {}).get("rag", [])
    web_results = result.get("evidence", {}).get("web", [])
    citations = result.get("citations", [])
    logs = [str(entry) for entry in result.get("log", [])]

    # --- ENRICHMENT STEP (With better image scraping) ---
    for w in web_results[:5]:
        if not w.get("image_url") or not w.get("rating"):
            extra_data = enrich_web_result(w.get("url"))
            w.update(extra_data)

    # 3. Format Products (Catalog)
    products_html = ""
    if products:
        products_html = "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px;'>"
        for p in products[:10]:
            img_html = ""
            imgs = p.get("image_urls", [])
            if isinstance(imgs, str): imgs = [imgs]
            
            if imgs and len(imgs) > 0:
                img_html = f"<img src='{imgs[0]}' style='width:100%; height:180px; object-fit:contain; background:#fff; border-radius:4px; margin-bottom:10px;'>"
            
            products_html += f"""
            <div style="background: #27272a; border: 1px solid #3f3f46; border-radius: 8px; padding: 15px; display: flex; flex-direction: column;">
                {img_html}
                <div style="font-weight: 600; color: #fff; margin-bottom: 5px; font-size: 1.05em;">{p.get('title', 'Unknown')}</div>
                <div style="color: #34d399; font-weight: bold; margin-bottom: 8px;">{p.get('price', 'N/A')}</div>
                <div style="font-size: 0.9em; color: #a1a1aa; line-height: 1.4; overflow-y: auto; max-height: 100px;">
                    {p.get('snippet', 'No description.')}
                </div>
            </div>
            """
        products_html += "</div>"
    else:
        products_html = "<p style='color:#a1a1aa;'>No catalog items found.</p>"

    # 4. Format Web Results (With Images)
    web_html = ""
    if web_results:
        web_html = "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px;'>"
        for w in web_results[:10]:
            # Image Logic for Web Results
            img_src = w.get("image_url") or w.get("thumbnail")
            img_tag = ""
            if img_src:
                img_tag = f"<img src='{img_src}' style='width:100%; height:150px; object-fit:cover; border-radius:4px; margin-bottom:10px; background:#18181b;'>"
            
            web_html += f"""
            <div style="background: #27272a; border: 1px solid #3f3f46; border-radius: 8px; padding: 15px;">
                {img_tag}
                <a href="{w.get('url','#')}" target="_blank" style="color: #818cf8; font-weight: 600; text-decoration: none; display: block; margin-bottom: 8px;">{w.get('title','Link')}</a>
                <div style="font-size: 0.9em; color: #d4d4d8; line-height: 1.5;">
                    {w.get('snippet', 'No snippet available.')}
                </div>
            </div>
            """
        web_html += "</div>"
    else:
        web_html = "<p style='color:#a1a1aa;'>No web results found.</p>"

    # 5. Generate AI Comparison
    comparison_html = llm_generate_comparison(products, web_results, final_query)

    # 6. Citations
    citations_html = ""
    if citations:
        citations_html = "<div style='background: #27272a; padding: 15px; border-radius: 8px;'>"
        for cite in citations:
            val = cite.get("doc_id") or cite.get("url")
            citations_html += f"<div style='color: #a1a1aa; border-bottom: 1px solid #3f3f46; padding: 5px 0;'>{val}</div>"
        citations_html += "</div>"

    # 7. Metrics
    metrics_html = f"""
    <div style="display: flex; gap: 20px; background: #18181b; padding: 15px; border-radius: 8px; border: 1px solid #3f3f46;">
        <div style="text-align: center; flex: 1;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #10b981;">{len(products)}</div>
            <div style="color: #a1a1aa; font-size: 0.8rem;">Catalog Items</div>
        </div>
        <div style="text-align: center; flex: 1;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #6366f1;">{len(web_results)}</div>
            <div style="color: #a1a1aa; font-size: 0.8rem;">Web Hits</div>
        </div>
    </div>
    """

    return transcription_update, answer, metrics_html, products_html, web_html, citations_html, comparison_html, logs

def generate_audio(answer):
    if not answer: return None
    clean_answer = re.sub(r'\(Sources?:.*?\)', '', answer).strip()
    try:
        return synthesize(clean_answer)
    except:
        return None

# --- UI Setup ---
with gr.Blocks(title="AI Product Assistant", theme=gr.themes.Monochrome(primary_hue="indigo")) as app:
    
    gr.HTML("""
    <div style="text-align: center; padding: 30px; background: linear-gradient(to right, #111827, #1f2937); border-radius: 0 0 15px 15px; margin-bottom: 20px;">
        <h1 style="color: white; font-weight: 800; font-size: 2rem;">AI Shopping Agent</h1>
        <p style="color: #9ca3af;">Internal Catalog vs. Live Web Comparison</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_method = gr.Radio(["Text", "Voice"], value="Text", label="Input Mode")
            text_input = gr.Textbox(label="Query", placeholder="e.g. Best noise cancelling headphones under $300", lines=2)
            audio_input = gr.Audio(label="Voice", type="filepath", visible=False)
            search_btn = gr.Button("Search", variant="primary", size="lg")
            
    with gr.Row():
        with gr.Column():
            # Transcription box is now dynamic (hidden by default, shows on voice input)
            transcription_output = gr.Textbox(label="Transcription", visible=False)
            answer_output = gr.Textbox(label="Agent Summary", lines=6, show_copy_button=True)
            metrics_output = gr.HTML(label="Stats")
            
    with gr.Tabs():
        with gr.Tab("Comparison"):
            comparison_output = gr.HTML()
        with gr.Tab("Catalog Products"):
            products_output = gr.HTML()
        with gr.Tab("Web Results"):
            web_output = gr.HTML()
        with gr.Tab("Citations"):
            citations_output = gr.HTML()
        with gr.Tab("Execution Logs"):
            logs_output = gr.JSON(label="Graph State Logs")
        with gr.Tab("Audio"):
            audio_btn = gr.Button("Generate Speech")
            audio_output = gr.Audio()

    # Interactivity
    def toggle(m): return (gr.update(visible=True), gr.update(visible=False)) if m == "Text" else (gr.update(visible=False), gr.update(visible=True))
    input_method.change(toggle, input_method, [text_input, audio_input])
    
    search_btn.click(
        process_query,
        [text_input, audio_input, input_method],
        [transcription_output, answer_output, metrics_output, products_output, web_output, citations_output, comparison_output, logs_output]
    )
    audio_btn.click(generate_audio, answer_output, audio_output)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=8888, share=True)