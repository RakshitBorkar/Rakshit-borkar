"""
ui/app.py
─────────
Gradio web interface for your SLM with RAG.
Supports: chat, document upload, RAG toggle, model settings.

Run:
    python ui/app.py
    python ui/app.py --model Qwen/Qwen2.5-3B-Instruct --adapter ./checkpoints/sft/final_adapter
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from loguru import logger


# ── Lazy model loading (avoid importing heavy deps at module level) ────────────
_engine = None
_rag    = None


def get_engine(model_name: str, adapter_path: Optional[str] = None):
    global _engine
    if _engine is None:
        from core.model import SLMEngine
        logger.info("Loading model...")
        _engine = SLMEngine(
            model_name_or_path=model_name,
            lora_adapter_path=adapter_path,
            quantize=True,
        )
    return _engine


def get_rag(use_rag: bool = True):
    global _rag
    if use_rag and _rag is None:
        from rag.pipeline import RAGPipeline
        _rag = RAGPipeline()
        logger.info(f"RAG loaded. Chunks in store: {_rag.stats()['total_chunks']}")
    return _rag if use_rag else None


# ── Chat logic ────────────────────────────────────────────────────────────────
def chat(
    message: str,
    history: List[Tuple[str, str]],
    model_name: str,
    adapter_path: str,
    use_rag: bool,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
):
    if not message.strip():
        return history, history, ""

    engine = get_engine(model_name, adapter_path or None)
    engine.temperature   = temperature
    engine.max_new_tokens = max_tokens

    sources_used = []

    if use_rag:
        rag = get_rag(True)
        if rag and rag.stats()["total_chunks"] > 0:
            prompt, sources = rag.build_rag_prompt(message, system_prompt)
            sources_used    = sources
        else:
            prompt = message
    else:
        prompt = message

    # Generate (non-streaming for simplicity)
    response = engine.generate(
        prompt=prompt,
        system_prompt=system_prompt if not use_rag else "",
    )

    if sources_used:
        source_names = [Path(s).name for s in sources_used]
        response += f"\n\n*Sources: {', '.join(source_names)}*"

    history.append((message, response))
    return history, history, ""


def upload_documents(files, progress=gr.Progress()):
    if not files:
        return "No files uploaded."

    from rag.pipeline import RAGPipeline, DocumentLoader
    rag = get_rag(True)

    results = []
    for i, file in enumerate(files):
        progress(i / len(files), desc=f"Processing {Path(file.name).name}...")
        try:
            doc = DocumentLoader.load_file(file.name)
            if doc:
                rag.ingest_document(doc)
                results.append(f"✅ {Path(file.name).name}")
            else:
                results.append(f"❌ {Path(file.name).name} (unsupported format)")
        except Exception as e:
            results.append(f"❌ {Path(file.name).name} ({e})")

    stats  = rag.stats()
    report = "\n".join(results)
    return f"{report}\n\n📚 Total chunks in store: {stats['total_chunks']}"


def ingest_url(url: str):
    if not url.strip():
        return "Please enter a URL."
    try:
        rag = get_rag(True)
        rag.ingest_url(url.strip())
        stats = rag.stats()
        return f"✅ Ingested: {url}\n📚 Total chunks: {stats['total_chunks']}"
    except Exception as e:
        return f"❌ Failed to ingest {url}: {e}"


def get_rag_stats():
    rag = get_rag(True)
    if rag:
        stats = rag.stats()
        return f"📚 Chunks in vector store: {stats['total_chunks']}"
    return "RAG not initialized."


# ── UI ────────────────────────────────────────────────────────────────────────
def build_ui(model_name: str, adapter_path: Optional[str]) -> gr.Blocks:

    with gr.Blocks(
        title="🧠 Your SLM",
        theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
        css="""
        #title { text-align: center; margin-bottom: 0; }
        #subtitle { text-align: center; color: #888; margin-top: 4px; }
        .source-badge { font-size: 11px; color: #666; }
        """
    ) as demo:

        gr.Markdown("# 🧠 Your SLM + RAG + RL", elem_id="title")
        gr.Markdown(
            "Your own self-improving language model with retrieval-augmented generation.",
            elem_id="subtitle"
        )

        with gr.Tabs():

            # ── Chat Tab ──────────────────────────────────────────────────────
            with gr.TabItem("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(height=500, label="Conversation")
                        with gr.Row():
                            msg_box = gr.Textbox(
                                placeholder="Ask anything...",
                                show_label=False,
                                scale=5,
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)
                        clear_btn = gr.Button("🗑️ Clear Chat", size="sm")

                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Settings")
                        model_input   = gr.Textbox(label="Model Name", value=model_name)
                        adapter_input = gr.Textbox(
                            label="LoRA Adapter Path",
                            value=adapter_path or "",
                            placeholder="./checkpoints/sft/final_adapter"
                        )
                        use_rag   = gr.Checkbox(label="🔍 Use RAG", value=True)
                        temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
                        max_tokens  = gr.Slider(64, 1024, value=512, step=64, label="Max Tokens")
                        sys_prompt  = gr.Textbox(
                            label="System Prompt",
                            value="You are a helpful, accurate, and concise AI assistant.",
                            lines=3,
                        )
                        rag_stats_btn = gr.Button("📊 RAG Stats", size="sm")
                        rag_stats_out = gr.Textbox(label="", lines=1, interactive=False)

                history_state = gr.State([])

                send_btn.click(
                    chat,
                    inputs=[msg_box, history_state, model_input, adapter_input,
                            use_rag, temperature, max_tokens, sys_prompt],
                    outputs=[chatbot, history_state, msg_box],
                )
                msg_box.submit(
                    chat,
                    inputs=[msg_box, history_state, model_input, adapter_input,
                            use_rag, temperature, max_tokens, sys_prompt],
                    outputs=[chatbot, history_state, msg_box],
                )
                clear_btn.click(lambda: ([], []), outputs=[chatbot, history_state])
                rag_stats_btn.click(get_rag_stats, outputs=rag_stats_out)

            # ── Documents Tab ─────────────────────────────────────────────────
            with gr.TabItem("📚 Documents"):
                gr.Markdown("### Upload documents to the RAG knowledge base")
                with gr.Row():
                    with gr.Column():
                        file_upload = gr.File(
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".md", ".docx"],
                            label="Upload Files (PDF, TXT, MD, DOCX)",
                        )
                        upload_btn = gr.Button("📥 Ingest Files", variant="primary")
                        upload_out = gr.Textbox(label="Result", lines=8, interactive=False)

                    with gr.Column():
                        gr.Markdown("### Or ingest from URL")
                        url_input = gr.Textbox(
                            label="URL",
                            placeholder="https://example.com/article",
                        )
                        url_btn = gr.Button("🌐 Ingest URL", variant="secondary")
                        url_out = gr.Textbox(label="Result", lines=4, interactive=False)

                upload_btn.click(upload_documents, inputs=[file_upload], outputs=[upload_out])
                url_btn.click(ingest_url, inputs=[url_input], outputs=[url_out])

            # ── Training Tab ─────────────────────────────────────────────────
            with gr.TabItem("🎓 Training Guide"):
                gr.Markdown("""
## Training Your SLM

### Step 1: Prepare Data
Your training data should be in JSONL format:
```json
{"instruction": "What is Python?", "input": "", "output": "Python is a programming language..."}
```
Place files in `./data/train.jsonl` and `./data/val.jsonl`

---

### Step 2: Supervised Fine-Tuning (SFT)
```bash
python training/sft_train.py --config configs/sft_config.yaml
```
This fine-tunes the base model on your domain data using LoRA (efficient, low VRAM).

---

### Step 3: RL Fine-Tuning (GRPO)
```bash
python rl/grpo_train.py --config configs/sft_config.yaml
```
Uses GRPO (same as DeepSeek-R1) to improve response quality via reward signals.

---

### Step 4: Self-Improvement Loop
```bash
python scripts/self_improve.py --rounds 5 --model ./checkpoints/sft/final_adapter
```
The model generates its own training data, scores it, and retrains itself iteratively.

---

### Hardware Requirements
| Model   | Min VRAM | Recommended |
|---------|----------|-------------|
| 1.5B    | 4GB      | 6GB         |
| 3B      | 8GB      | 12GB        |
| 7B      | 16GB     | 24GB        |

*With 4-bit quantization enabled (default)*
                """)

        gr.Markdown(
            "Built with ❤️ | Qwen2.5 + LoRA + GRPO + ChromaDB",
            elem_id="subtitle"
        )

    return demo


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--port",    type=int, default=7860)
    parser.add_argument("--share",   action="store_true")
    args = parser.parse_args()

    demo = build_ui(args.model, args.adapter)
    demo.launch(server_port=args.port, share=args.share, server_name="0.0.0.0")