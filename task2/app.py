"""
EE471 In-Class Week 10 – Task 2
Gradio interactive demo for all 10 NLP tasks.

Each model runs in a dedicated worker PROCESS (not thread) so the GIL
in worker processes never blocks the Gradio server process.
"""

import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import gradio as gr
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Module-level worker functions (must be top-level to be picklable by spawn)
# Models are cached inside each worker process after the first call.
# ─────────────────────────────────────────────────────────────────────────────

_cache: dict = {}


def _w_sentiment(text):
    if "sa" not in _cache:
        from nlp_module import SentimentAnalyzer
        _cache["sa"] = SentimentAnalyzer()
    return _cache["sa"].run(text)


def _w_zero_shot(text, labels):
    if "zsc" not in _cache:
        from nlp_module import ZeroShotClassifier
        _cache["zsc"] = ZeroShotClassifier()
    return _cache["zsc"].run(text, labels)


def _w_text_gen(prompt, max_new_tokens, num_sequences):
    if "tg" not in _cache:
        from nlp_module import TextGenerator
        _cache["tg"] = TextGenerator()
    return _cache["tg"].run(prompt, max_new_tokens=max_new_tokens, num_sequences=num_sequences)


def _w_mask_fill(masked_text, top_k):
    if "mf" not in _cache:
        from nlp_module import MaskFiller
        _cache["mf"] = MaskFiller()
    return _cache["mf"].run(masked_text, top_k=top_k)


def _w_ner(text):
    if "ner" not in _cache:
        from nlp_module import NERTagger
        _cache["ner"] = NERTagger()
    return _cache["ner"].run(text)


def _w_qa(question, context):
    if "qa" not in _cache:
        from nlp_module import QuestionAnswerer
        _cache["qa"] = QuestionAnswerer()
    return _cache["qa"].run(question, context)


def _w_summarize(text, max_length, min_length):
    if "sum" not in _cache:
        from nlp_module import TextSummarizer
        _cache["sum"] = TextSummarizer()
    return _cache["sum"].run(text, max_length=max_length, min_length=min_length)


def _w_translate(text):
    if "tr" not in _cache:
        from nlp_module import Translator
        _cache["tr"] = Translator()
    return _cache["tr"].run(text)


def _w_image_classify(image_bytes):
    import io
    if "img" not in _cache:
        from nlp_module import ImageClassifier
        _cache["img"] = ImageClassifier()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _cache["img"].run(image, top_k=5)


def _w_asr(audio_path):
    if "asr" not in _cache:
        from nlp_module import SpeechRecognizer
        _cache["asr"] = SpeechRecognizer()
    return _cache["asr"].run(audio_path)


# ─────────────────────────────────────────────────────────────────────────────
# Process pool — created only in the main process
# ─────────────────────────────────────────────────────────────────────────────

_pool: ProcessPoolExecutor | None = None


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None and mp.current_process().name == "MainProcess":
        _pool = ProcessPoolExecutor(max_workers=10)
    return _pool  # type: ignore[return-value]


async def _run(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_get_pool(), fn, *args)


LOADING = "⏳ Model loading for the first time, please wait…"


# ─────────────────────────────────────────────────────────────────────────────
# Async Gradio handlers
# ─────────────────────────────────────────────────────────────────────────────

async def run_sentiment(text):
    if not text.strip():
        yield "Please enter some text."; return
    yield LOADING
    result = await _run(_w_sentiment, text)
    emoji = "🟢" if result["label"].upper() == "POSITIVE" else "🔴"
    yield f"{emoji} **{result['label']}**  (confidence: {result['score']:.4f})"


async def run_zero_shot(text, labels_raw):
    if not text.strip() or not labels_raw.strip():
        yield "Please enter text and labels."; return
    labels = [l.strip() for l in labels_raw.split(",") if l.strip()]
    if not labels:
        yield "Please enter comma-separated labels."; return
    yield LOADING
    result = await _run(_w_zero_shot, text, labels)
    lines = [f"🏆 **Top label:** {result['top_label']}\n"]
    lines += [f"- `{s['label']}`: {s['score']:.4f}" for s in result["scores"]]
    yield "\n".join(lines)


async def run_text_gen(prompt, max_tokens, num_seqs):
    if not prompt.strip():
        yield "Please enter a prompt."; return
    yield LOADING
    completions = await _run(_w_text_gen, prompt, int(max_tokens), int(num_seqs))
    yield "\n---\n".join(f"**Alternative {i}:**\n{t}\n" for i, t in enumerate(completions, 1))


async def run_mask_fill(masked_text, top_k):
    if not masked_text.strip():
        yield "Please enter a sentence with [MASK]."; return
    if "[MASK]" not in masked_text:
        yield "Your sentence must contain the `[MASK]` token."; return
    yield LOADING
    results = await _run(_w_mask_fill, masked_text, int(top_k))
    yield "\n".join(f"- **[{r['token']}]** ({r['score']:.4f}) → _{r['sequence']}_" for r in results)


async def run_ner(text):
    if not text.strip():
        yield "Please enter some text."; return
    yield LOADING
    entities = await _run(_w_ner, text)
    if not entities:
        yield "No named entities found."; return
    lines = ["| Entity | Word | Score |", "|---|---|---|"]
    lines += [f"| `{e['entity']}` | {e['word']} | {e['score']:.4f} |" for e in entities]
    yield "\n".join(lines)


async def run_qa(question, context):
    if not question.strip() or not context.strip():
        yield "Please enter both a question and a context passage."; return
    yield LOADING
    result = await _run(_w_qa, question, context)
    yield f"**Answer:** {result['answer']}\n\n**Confidence:** {result['score']:.4f}"


async def run_summarize(text, max_len, min_len):
    if not text.strip():
        yield "Please enter text to summarise."; return
    if len(text.split()) < 30:
        yield "Please enter a longer text (at least ~30 words)."; return
    yield "⏳ Generating summary, please wait…"
    yield await _run(_w_summarize, text, int(max_len), int(min_len))


async def run_translate(text):
    if not text.strip():
        yield "Please enter English text to translate."; return
    yield "⏳ Translating, please wait…"
    yield await _run(_w_translate, text)


async def run_image_classify(image: np.ndarray | None):
    if image is None:
        yield "Please upload an image."; return
    yield LOADING
    import io
    buf = io.BytesIO()
    Image.fromarray(image.astype("uint8"), "RGB").save(buf, format="PNG")
    results = await _run(_w_image_classify, buf.getvalue())
    lines = ["| Rank | Label | Score |", "|---|---|---|"]
    lines += [f"| {i} | {r['label']} | {r['score']:.4f} |" for i, r in enumerate(results, 1)]
    yield "\n".join(lines)


async def run_asr(audio_path):
    if audio_path is None:
        yield "Please upload or record an audio file."; return
    yield "⏳ Transcribing, please wait…"
    yield await _run(_w_asr, audio_path)


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="EE471 NLP Demo") as demo:
    gr.Markdown("# 🤗 NLP Task Demo — EE471 Week 10")
    gr.Markdown("Interactive demos for **10 NLP tasks**. Each task runs in its own background process — UI never freezes.")

    with gr.Tabs(elem_id="main_tabs"):

        with gr.Tab("1 · Sentiment Analysis", id="tab_1"):
            gr.Markdown("Classify text as POSITIVE / NEGATIVE · `siebert/sentiment-roberta-large-english`")
            with gr.Row():
                sa_in  = gr.Textbox(label="Input Text", placeholder="I've been waiting for this course my whole life.", lines=3)
                sa_out = gr.Markdown(label="Result")
            sa_btn = gr.Button("Analyse", variant="primary")
            sa_btn.click(run_sentiment, inputs=sa_in, outputs=sa_out, concurrency_id="sa")
            gr.Examples([["I've been waiting for a EE471 course my whole life."], ["I hate this assignment."]], inputs=sa_in)

        with gr.Tab("2 · Zero-Shot Classification", id="tab_2"):
            gr.Markdown("Classify into custom labels · `facebook/bart-large-mnli`")
            zsc_text   = gr.Textbox(label="Input Text", placeholder="Berkshire keeps their cash reserves at an extremely high level.", lines=3)
            zsc_labels = gr.Textbox(label="Candidate Labels (comma-separated)", placeholder="finance, technology, sports, politics, health")
            zsc_out    = gr.Markdown(label="Result")
            zsc_btn    = gr.Button("Classify", variant="primary")
            zsc_btn.click(run_zero_shot, inputs=[zsc_text, zsc_labels], outputs=zsc_out, concurrency_id="zsc")

        with gr.Tab("3 · Text Generation", id="tab_3"):
            gr.Markdown("Complete an incomplete sentence · `gpt2`")
            tg_prompt = gr.Textbox(label="Prompt", placeholder="If I continue to successfully complete all in-class exercises,", lines=3)
            with gr.Row():
                tg_tokens = gr.Slider(10, 100, value=50, step=5, label="Max New Tokens")
                tg_seqs   = gr.Slider(1, 4, value=2, step=1, label="Alternatives")
            tg_out = gr.Markdown(label="Completions")
            tg_btn = gr.Button("Generate", variant="primary")
            tg_btn.click(run_text_gen, inputs=[tg_prompt, tg_tokens, tg_seqs], outputs=tg_out, concurrency_id="tg")

        with gr.Tab("4 · Mask Filling", id="tab_4"):
            gr.Markdown("Predict `[MASK]` token · `bert-base-uncased`")
            mf_in   = gr.Textbox(label="Masked Sentence", placeholder="To understand generative AI, one must study [MASK] well.", lines=3)
            mf_topk = gr.Slider(1, 10, value=5, step=1, label="Top-K")
            mf_out  = gr.Markdown(label="Predictions")
            mf_btn  = gr.Button("Fill Mask", variant="primary")
            mf_btn.click(run_mask_fill, inputs=[mf_in, mf_topk], outputs=mf_out, concurrency_id="mf")

        with gr.Tab("5 · Named Entity Recognition", id="tab_5"):
            gr.Markdown("Extract persons, orgs, locations · default NER model")
            ner_in  = gr.Textbox(label="Input Text", placeholder="I am Nate, a research assistant in Izmir Institute of Technology, living in İzmir, Türkiye.", lines=4)
            ner_out = gr.Markdown(label="Extracted Entities")
            ner_btn = gr.Button("Recognise Entities", variant="primary")
            ner_btn.click(run_ner, inputs=ner_in, outputs=ner_out, concurrency_id="ner")

        with gr.Tab("6 · Question Answering", id="tab_6"):
            gr.Markdown("Extract answer from context · `deepset/roberta-base-squad2`")
            qa_ctx = gr.Textbox(label="Context Passage", placeholder="Paste your text here…", lines=6)
            qa_q   = gr.Textbox(label="Question", placeholder="What is the person's name?", lines=2)
            qa_out = gr.Markdown(label="Answer")
            qa_btn = gr.Button("Answer", variant="primary")
            qa_btn.click(run_qa, inputs=[qa_q, qa_ctx], outputs=qa_out, concurrency_id="qa")

        with gr.Tab("7 · Text Summarization", id="tab_7"):
            gr.Markdown("Abstractive summary · `facebook/bart-large-cnn`")
            sum_in = gr.Textbox(label="Long Text", placeholder="Paste a long article here…", lines=8)
            with gr.Row():
                sum_max = gr.Slider(40, 200, value=80, step=10, label="Max Length")
                sum_min = gr.Slider(10, 100, value=25, step=5,  label="Min Length")
            sum_out = gr.Textbox(label="Summary", lines=4, interactive=False)
            sum_btn = gr.Button("Summarise", variant="primary")
            sum_btn.click(run_summarize, inputs=[sum_in, sum_max, sum_min], outputs=sum_out, concurrency_id="sum")

        with gr.Tab("8 · Text Translation (EN → TR)", id="tab_8"):
            gr.Markdown("Translate to Turkish · `facebook/nllb-200-distilled-600M`")
            tr_in  = gr.Textbox(label="English Text", placeholder="The 2008 financial crisis was the most severe collapse of the 21st century.", lines=4)
            tr_out = gr.Textbox(label="Turkish Translation", lines=4, interactive=False)
            tr_btn = gr.Button("Translate", variant="primary")
            tr_btn.click(run_translate, inputs=tr_in, outputs=tr_out, concurrency_id="tr")

        with gr.Tab("9 · Image Classification", id="tab_9"):
            gr.Markdown("Classify images · `google/vit-base-patch16-224`")
            with gr.Row():
                img_in  = gr.Image(label="Upload Image", type="numpy")
                img_out = gr.Markdown(label="Top-5 Predictions")
            img_btn = gr.Button("Classify Image", variant="primary")
            img_btn.click(run_image_classify, inputs=img_in, outputs=img_out, concurrency_id="img")

        with gr.Tab("10 · Speech Recognition", id="tab_10"):
            gr.Markdown("Transcribe speech · `openai/whisper-large-v3`")
            asr_in  = gr.Audio(label="Audio Input", sources=["upload"], type="filepath")
            asr_out = gr.Textbox(label="Transcription", lines=4, interactive=False)
            asr_btn = gr.Button("Transcribe", variant="primary")
            asr_btn.click(run_asr, inputs=asr_in, outputs=asr_out, concurrency_id="asr")

    gr.Markdown("---\n*EE471 – In-Class Exercise Week 10 · HuggingFace Transformers*")


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=10)
    demo.launch(share=False, theme=gr.themes.Soft(), max_threads=200)
