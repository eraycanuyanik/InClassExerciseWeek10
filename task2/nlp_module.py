"""
EE471 In-Class Week 10 – Task 2
OOP NLP Module wrapping HuggingFace Transformers pipelines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import io
import tempfile
import os

import numpy as np
import requests
import torch
import librosa
from PIL import Image

from transformers import (
    pipeline,
    BartForConditionalGeneration,
    BartTokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

class NLPTask(ABC):
    """Abstract base class for all NLP tasks."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()
            self._loaded = True

    @abstractmethod
    def _load(self) -> None:
        """Load model/pipeline (called lazily on first run)."""

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the task and return results."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ──────────────────────────────────────────────────────────────────────────────
# Task 1 – Sentiment Analysis
# ──────────────────────────────────────────────────────────────────────────────

class SentimentAnalyzer(NLPTask):
    """Classifies text as POSITIVE or NEGATIVE with a confidence score."""

    def __init__(self, model: str = "siebert/sentiment-roberta-large-english") -> None:
        super().__init__("Sentiment Analysis")
        self.model = model
        self._pipeline = None

    def _load(self) -> None:
        self._pipeline = pipeline("sentiment-analysis", model=self.model)

    def run(self, text: str) -> dict[str, Any]:
        """
        Parameters
        ----------
        text : str  Input sentence or paragraph.

        Returns
        -------
        dict with keys: label (str), score (float)
        """
        self._ensure_loaded()
        result = self._pipeline(text)[0]
        return {"label": result["label"], "score": round(result["score"], 4)}


# ──────────────────────────────────────────────────────────────────────────────
# Task 2 – Zero-Shot Classification
# ──────────────────────────────────────────────────────────────────────────────

class ZeroShotClassifier(NLPTask):
    """Classifies text against arbitrary candidate labels without fine-tuning."""

    def __init__(self, model: str = "facebook/bart-large-mnli") -> None:
        super().__init__("Zero-Shot Classification")
        self.model = model
        self._pipeline = None

    def _load(self) -> None:
        self._pipeline = pipeline("zero-shot-classification", model=self.model)

    def run(self, text: str, labels: list[str]) -> dict[str, Any]:
        """
        Parameters
        ----------
        text   : str         Input sentence.
        labels : list[str]   Candidate label names.

        Returns
        -------
        dict with keys: top_label (str), scores (list of {label, score})
        """
        self._ensure_loaded()
        result = self._pipeline(text, candidate_labels=labels)
        scores = [
            {"label": lbl, "score": round(sc, 4)}
            for lbl, sc in zip(result["labels"], result["scores"])
        ]
        return {"top_label": result["labels"][0], "scores": scores}


# ──────────────────────────────────────────────────────────────────────────────
# Task 3 – Text Generation
# ──────────────────────────────────────────────────────────────────────────────

class TextGenerator(NLPTask):
    """Completes an incomplete sentence using a causal language model."""

    def __init__(self, model: str = "gpt2") -> None:
        super().__init__("Text Generation")
        self.model = model
        self._pipeline = None

    def _load(self) -> None:
        self._pipeline = pipeline("text-generation", model=self.model)

    def run(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        num_sequences: int = 2,
    ) -> list[str]:
        """
        Parameters
        ----------
        prompt         : str  Incomplete sentence to continue.
        max_new_tokens : int  Tokens to generate beyond the prompt.
        num_sequences  : int  How many alternative completions to return.

        Returns
        -------
        list of generated strings
        """
        self._ensure_loaded()
        results = self._pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_sequences,
            truncation=True,
        )
        return [r["generated_text"] for r in results]


# ──────────────────────────────────────────────────────────────────────────────
# Task 4 – Mask Filling
# ──────────────────────────────────────────────────────────────────────────────

class MaskFiller(NLPTask):
    """Predicts the most likely token(s) for a [MASK] placeholder."""

    def __init__(self, model: str = "bert-base-uncased") -> None:
        super().__init__("Mask Filling")
        self.model = model
        self._pipeline = None

    def _load(self) -> None:
        self._pipeline = pipeline("fill-mask", model=self.model)

    def run(self, masked_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Parameters
        ----------
        masked_text : str  Sentence containing exactly one [MASK] token.
        top_k       : int  Number of candidates to return.

        Returns
        -------
        list of dicts with keys: token (str), score (float), sequence (str)
        """
        self._ensure_loaded()
        results = self._pipeline(masked_text)
        return [
            {
                "token": r["token_str"].strip(),
                "score": round(r["score"], 4),
                "sequence": r["sequence"],
            }
            for r in results[:top_k]
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Task 5 – Named Entity Recognition
# ──────────────────────────────────────────────────────────────────────────────

class NERTagger(NLPTask):
    """Extracts named entities (persons, organisations, locations, etc.)."""

    def __init__(self, model: str | None = None) -> None:
        super().__init__("Named Entity Recognition")
        self.model = model
        self._pipeline = None

    def _load(self) -> None:
        kwargs: dict[str, Any] = {"aggregation_strategy": "simple"}
        if self.model:
            kwargs["model"] = self.model
        self._pipeline = pipeline("ner", **kwargs)

    def run(self, text: str) -> list[dict[str, Any]]:
        """
        Parameters
        ----------
        text : str  Input sentence or paragraph.

        Returns
        -------
        list of dicts with keys: entity (str), word (str), score (float)
        """
        self._ensure_loaded()
        results = self._pipeline(text)
        return [
            {
                "entity": r["entity_group"],
                "word": r["word"],
                "score": round(r["score"], 4),
            }
            for r in results
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Task 6 – Question Answering
# ──────────────────────────────────────────────────────────────────────────────

class QuestionAnswerer(NLPTask):
    """Extracts an answer span from a given context passage."""

    def __init__(self, model: str = "deepset/roberta-base-squad2") -> None:
        super().__init__("Question Answering")
        self.model = model
        self._pipeline = None

    def _load(self) -> None:
        self._pipeline = pipeline("question-answering", model=self.model)

    def run(self, question: str, context: str) -> dict[str, Any]:
        """
        Parameters
        ----------
        question : str  Natural-language question.
        context  : str  Passage that contains the answer.

        Returns
        -------
        dict with keys: answer (str), score (float), start (int), end (int)
        """
        self._ensure_loaded()
        result = self._pipeline(question=question, context=context)
        return {
            "answer": result["answer"],
            "score": round(result["score"], 4),
            "start": result["start"],
            "end": result["end"],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Task 7 – Text Summarization
# ──────────────────────────────────────────────────────────────────────────────

class TextSummarizer(NLPTask):
    """Generates an abstractive summary using facebook/bart-large-cnn."""

    def __init__(self, model: str = "facebook/bart-large-cnn") -> None:
        super().__init__("Text Summarization")
        self.model = model
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        self._tokenizer = BartTokenizer.from_pretrained(self.model)
        self._model = BartForConditionalGeneration.from_pretrained(self.model)

    def run(
        self,
        text: str,
        max_length: int = 80,
        min_length: int = 25,
    ) -> str:
        """
        Parameters
        ----------
        text       : str  Long input text to summarise.
        max_length : int  Maximum summary token length.
        min_length : int  Minimum summary token length.

        Returns
        -------
        Summary string.
        """
        self._ensure_loaded()
        inputs = self._tokenizer(
            text, return_tensors="pt", max_length=1024, truncation=True
        )
        with torch.no_grad():
            summary_ids = self._model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )
        return self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# ──────────────────────────────────────────────────────────────────────────────
# Task 8 – Text Translation (EN → TR)
# ──────────────────────────────────────────────────────────────────────────────

class Translator(NLPTask):
    """Translates English text to Turkish using facebook/nllb-200-distilled-600M."""

    def __init__(
        self,
        model: str = "facebook/nllb-200-distilled-600M",
        src_lang: str = "eng_Latn",
        tgt_lang: str = "tur_Latn",
    ) -> None:
        super().__init__("Text Translation")
        self.model = model
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self._tokenizer = None
        self._model = None
        self._tgt_token_id = None

    def _load(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model)
        self._tgt_token_id = self._tokenizer.convert_tokens_to_ids(self.tgt_lang)

    def run(self, text: str) -> str:
        """
        Parameters
        ----------
        text : str  English input text.

        Returns
        -------
        Turkish translation string.
        """
        self._ensure_loaded()
        inputs = self._tokenizer(text, return_tensors="pt", src_lang=self.src_lang)
        with torch.no_grad():
            tokens = self._model.generate(
                **inputs,
                forced_bos_token_id=self._tgt_token_id,
                max_length=200,
                num_beams=5,
                no_repeat_ngram_size=3,
            )
        return self._tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]


# ──────────────────────────────────────────────────────────────────────────────
# Task 9 – Image Classification
# ──────────────────────────────────────────────────────────────────────────────

class ImageClassifier(NLPTask):
    """Classifies images using google/vit-base-patch16-224."""

    def __init__(self, model: str = "google/vit-base-patch16-224") -> None:
        super().__init__("Image Classification")
        self.model = model
        self._pipeline = None

    def _load(self) -> None:
        self._pipeline = pipeline("image-classification", model=self.model)

    def run(self, image: Image.Image, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Parameters
        ----------
        image : PIL.Image.Image  Input image (RGB).
        top_k : int              Number of top predictions to return.

        Returns
        -------
        list of dicts with keys: label (str), score (float)
        """
        self._ensure_loaded()
        results = self._pipeline(image)
        return [
            {"label": r["label"], "score": round(r["score"], 4)}
            for r in results[:top_k]
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Task 10 – Automatic Speech Recognition
# ──────────────────────────────────────────────────────────────────────────────

class SpeechRecognizer(NLPTask):
    """Transcribes audio using openai/whisper-large-v3."""

    def __init__(self, model: str = "openai/whisper-large-v3") -> None:
        super().__init__("Automatic Speech Recognition")
        self.model = model
        self._processor = None
        self._model = None

    def _load(self) -> None:
        self._processor = WhisperProcessor.from_pretrained(self.model)
        self._model = WhisperForConditionalGeneration.from_pretrained(
            self.model, torch_dtype=torch.float32
        )
        self._model.eval()

    def run(self, audio_path: str, language: str = "en") -> str:
        """
        Parameters
        ----------
        audio_path : str  Path to a local audio file (any format librosa supports).
        language   : str  Hint language code (e.g. 'en', 'tr').

        Returns
        -------
        Transcription string.
        """
        self._ensure_loaded()
        audio_data, _ = librosa.load(audio_path, sr=16000)
        input_features = self._processor(
            audio_data, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(torch.float32)
        with torch.no_grad():
            predicted_ids = self._model.generate(
                input_features, language=language
            )
        return self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
