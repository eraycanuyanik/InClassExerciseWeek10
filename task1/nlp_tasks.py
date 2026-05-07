"""
EE471 In-Class Week 10 – HuggingFace Transformers Pipeline Tasks
"""

from transformers import (
    pipeline,
    BartForConditionalGeneration, BartTokenizer,
    AutoModelForSeq2SeqLM, AutoTokenizer,
    WhisperProcessor, WhisperForConditionalGeneration,
)
import torch
import tempfile
import os
import librosa
import numpy as np
import requests
from PIL import Image
import io

DIVIDER = "=" * 65


def section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ──────────────────────────────────────────────────────────────────
# Task 1 – Sentiment Analysis
# ──────────────────────────────────────────────────────────────────
section("TASK 1 – Sentiment Analysis")

# siebert/sentiment-roberta-large-english correctly classifies
# context-dependent positive phrases like "I've been waiting for X my whole life."
sentiment = pipeline(
    "sentiment-analysis",
    model="siebert/sentiment-roberta-large-english",
)
sentences = [
    "I've been waiting for a EE471 course my whole life.",
    "I hate EE471 course",
]
results = sentiment(sentences)
for sent, res in zip(sentences, results):
    print(f"  Text   : {sent}")
    print(f"  Label  : {res['label']}  |  Score : {res['score']:.4f}\n")


# ──────────────────────────────────────────────────────────────────
# Task 2 – Zero-Shot Classification
# ──────────────────────────────────────────────────────────────────
section("TASK 2 – Zero-Shot Classification")

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
zsc_text = "Berkshire keeps their cash reserves at an extremely high level."
candidate_labels = ["finance", "technology", "sports", "politics", "health"]
zsc_result = classifier(zsc_text, candidate_labels=candidate_labels)
print(f"  Text   : {zsc_text}")
print(f"  Labels : {candidate_labels}")
print(f"  Top    : {zsc_result['labels'][0]}  (score {zsc_result['scores'][0]:.4f})")
print("  All scores:")
for label, score in zip(zsc_result["labels"], zsc_result["scores"]):
    print(f"    {label:<12} → {score:.4f}")


# ──────────────────────────────────────────────────────────────────
# Task 3 – Text Generation (Sentence Completion)
# ──────────────────────────────────────────────────────────────────
section("TASK 3 – Text Generation (Sentence Completion)")

generator = pipeline("text-generation", model="gpt2")
prompt = "If I continue to successfully complete all in-class exercises in EE471 course,"
gen_results = generator(
    prompt,
    max_new_tokens=35,
    num_return_sequences=2,
    truncation=True,
)
print(f"  Prompt : {prompt}\n")
for i, gen in enumerate(gen_results, 1):
    print(f"  Alternative {i}:")
    print(f"    {gen['generated_text']}\n")


# ──────────────────────────────────────────────────────────────────
# Task 4 – Fill-Mask
# ──────────────────────────────────────────────────────────────────
section("TASK 4 – Fill-Mask")

unmasker = pipeline("fill-mask", model="bert-base-uncased")
masked_sentence = "To understand generative AI, one must study [MASK] well."
mask_results = unmasker(masked_sentence)
print(f"  Masked sentence : {masked_sentence}\n")
for res in mask_results[:5]:
    print(f"    [{res['token_str']:>15}]  score={res['score']:.4f}  → {res['sequence']}")


# ──────────────────────────────────────────────────────────────────
# Task 5 – Named Entity Recognition (NER)
# ──────────────────────────────────────────────────────────────────
section("TASK 5 – Named Entity Recognition (NER)")

ner = pipeline("ner", aggregation_strategy="simple")
ner_text = (
    "I am Nate, a research assistant in Izmir Institute of Technology, "
    "and currently living and working in beautiful city İzmir in Türkiye."
)
ner_results = ner(ner_text)
print(f"  Text: {ner_text}\n")
print("  Extracted entities:")
for entity in ner_results:
    print(f"    [{entity['entity_group']:>4}]  '{entity['word']}'  (score {entity['score']:.4f})")

# Collect key info for QA validation
extracted_person = next((e["word"] for e in ner_results if e["entity_group"] == "PER"), "Nate")
extracted_org    = next((e["word"] for e in ner_results if e["entity_group"] == "ORG"), "Izmir Institute of Technology")
extracted_loc    = next((e["word"] for e in ner_results if e["entity_group"] == "LOC"), "İzmir")

print(f"\n  → Person  : {extracted_person}")
print(f"  → Org     : {extracted_org}")
print(f"  → Location: {extracted_loc}")


# ──────────────────────────────────────────────────────────────────
# Task 6 – Question Answering (validate NER results)
# ──────────────────────────────────────────────────────────────────
section("TASK 6 – Question Answering (NER Validation)")

qa = pipeline("question-answering",
              model="deepset/roberta-base-squad2")
context = ner_text
questions = [
    ("What is the person's name?",         "Person"),
    ("Which organization does Nate work for?", "Organization"),
    ("Which city does Nate live in?",      "City"),
]
for question, label in questions:
    answer = qa(question=question, context=context)
    print(f"  Q: {question}")
    print(f"  A [{label}]: {answer['answer']}  (score {answer['score']:.4f})\n")


# ──────────────────────────────────────────────────────────────────
# Task 7 – Text Summarization
# ──────────────────────────────────────────────────────────────────
section("TASK 7 – Text Summarization")

long_text = (
    "The 2008 Global Financial Crisis stands as the most severe economic collapse of the 21st "
    "century, often compared to the Great Depression of the 1930s. Triggered by the bursting of "
    "the United States housing bubble, its effects rippled across the globe, leading to the "
    "collapse of major financial institutions and a deep international recession. The crisis began "
    "with the subprime mortgage market. In the early 2000s, low interest rates and a push for "
    "homeownership led banks to issue high-risk loans to borrowers with poor credit."
)

# In Transformers 5.x the "summarization" pipeline alias was removed;
# we load facebook/bart-large-cnn directly via its seq2seq API.
bart_model_name = "facebook/bart-large-cnn"
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)

inputs = bart_tokenizer(
    long_text, return_tensors="pt", max_length=1024, truncation=True
)
with torch.no_grad():
    summary_ids = bart_model.generate(
        inputs["input_ids"],
        max_length=80,
        min_length=25,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
summary_text = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(f"  Original ({len(long_text.split())} words):\n  {long_text}\n")
print(f"  Summary:\n  {summary_text}")


# ──────────────────────────────────────────────────────────────────
# Task 8 – Translation (English → Turkish)
# ──────────────────────────────────────────────────────────────────
section("TASK 8 – Translation (EN → TR)")

# facebook/nllb-200-distilled-600M is a state-of-the-art multilingual model
# that produces more natural Turkish than MarianMT for this sentence.
nllb_model_name = "facebook/nllb-200-distilled-600M"
nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name)

translate_text = (
    "The 2008 Global Financial Crisis stands as the most severe economic collapse "
    "of the 21st century, often compared to the Great Depression."
)
# Setting src_lang explicitly ensures the tokenizer adds the correct
# language token so the model knows the input is English, producing
# more fluent and grammatically complete Turkish output.
inputs = nllb_tokenizer(translate_text, return_tensors="pt", src_lang="eng_Latn")
tgt_token_id = nllb_tokenizer.convert_tokens_to_ids("tur_Latn")
with torch.no_grad():
    translated_tokens = nllb_model.generate(
        **inputs,
        forced_bos_token_id=tgt_token_id,
        max_length=200,
        num_beams=5,
        no_repeat_ngram_size=3,
    )
translated_text = nllb_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

print(f"  Original : {translate_text}")
print(f"  Turkish  : {translated_text}")


# ──────────────────────────────────────────────────────────────────
# Task 9 – Image Classification (ViT)
# ──────────────────────────────────────────────────────────────────
section("TASK 9 – Image Classification (google/vit-base-patch16-224)")

image_classifier = pipeline("image-classification",
                             model="google/vit-base-patch16-224")

# HuggingFace documentation sample image — reliably accessible
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
response = requests.get(image_url, timeout=30)
response.raise_for_status()
image = Image.open(io.BytesIO(response.content))

img_results = image_classifier(image)
print(f"  Image URL : {image_url}")
print("  Top-5 predictions:")
for pred in img_results[:5]:
    print(f"    {pred['label']:<40}  score={pred['score']:.4f}")


# ──────────────────────────────────────────────────────────────────
# Task 10 – Automatic Speech Recognition (Whisper)
# ──────────────────────────────────────────────────────────────────
section("TASK 10 – Automatic Speech Recognition (openai/whisper-large-v3)")

# Transformers 5.x ASR pipeline has a known bug with 'num_frames';
# we use WhisperProcessor + WhisperForConditionalGeneration directly.
whisper_name = "openai/whisper-large-v3"
whisper_processor = WhisperProcessor.from_pretrained(whisper_name)
whisper_model = WhisperForConditionalGeneration.from_pretrained(
    whisper_name, torch_dtype=torch.float32
)
whisper_model.eval()

# HuggingFace official sample FLAC (MLK speech excerpt).
# librosa resamples to 16 kHz — Whisper's required sampling rate.
audio_url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
audio_response = requests.get(audio_url, timeout=60)
audio_response.raise_for_status()

tmp_audio = tempfile.NamedTemporaryFile(suffix=".flac", delete=False)
try:
    tmp_audio.write(audio_response.content)
    tmp_audio.close()
    audio_data, _ = librosa.load(tmp_audio.name, sr=16000)
finally:
    os.unlink(tmp_audio.name)

input_features = whisper_processor(
    audio_data, sampling_rate=16000, return_tensors="pt"
).input_features.to(torch.float32)

with torch.no_grad():
    predicted_ids = whisper_model.generate(input_features, language="en")

transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print(f"  Audio URL  : {audio_url}")
print(f"  Transcript : {transcription[:500]}{'...' if len(transcription) > 500 else ''}")

print(f"\n{DIVIDER}")
print("  All tasks completed successfully!")
print(DIVIDER)
