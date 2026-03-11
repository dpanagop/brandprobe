from textblob import TextBlob
from typing import Optional
from transformers import pipeline
from .engines import BaseEngine
import re

_ROBERTA_PIPELINE = None

def _get_roberta_pipeline():
    """Lazily initializes and returns the RoBERTa sentiment pipeline."""
    global _ROBERTA_PIPELINE
    if _ROBERTA_PIPELINE is None:
        _ROBERTA_PIPELINE = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    return _ROBERTA_PIPELINE

class SentimentWrapper:
    @staticmethod
    def analyze(text: str, engine: Optional[BaseEngine] = None, method: str = "textblob") -> float:
        """
        Returns polarity score of the text ranging from -1.0 to 1.0.
        If `engine` is provided (and method is 'llm'), uses the LLM to score the sentiment.
        Otherwise, uses the specified `method` ('textblob' or 'roberta').
        """
        if not text:
            return 0.0
            
        if method == "llm" and engine is not None:
            return SentimentWrapper._analyze_llm(text, engine)
        elif method == "roberta":
            return SentimentWrapper._analyze_roberta(text)
        else:
            return SentimentWrapper._analyze_textblob(text)
            
    @staticmethod
    def _analyze_textblob(text: str) -> float:
        blob = TextBlob(text)
        return blob.sentiment.polarity
        
    @staticmethod
    def _analyze_roberta(text: str) -> float:
        pipe = _get_roberta_pipeline()
        # Truncate to avoid exceeding max length
        result = pipe(text[:512])[0]
        label = result['label'].lower()
        score = result['score']
        
        # typically labels are 'positive', 'neutral', 'negative'
        # cardiffnlp/twitter-roberta-base-sentiment outputs 'positive', 'neutral', 'negative'
        if label == 'positive':
            return float(score)  # 0.0 to 1.0
        elif label == 'negative':
            return float(-score) # -1.0 to 0.0
        else:
            return 0.0           # neutral
        
    @staticmethod
    def _analyze_llm(text: str, engine: BaseEngine) -> float:
        system_prompt = (
            "You are a sentiment analysis expert. Evaluate the sentiment of the provided text. "
            "Reply strictly with a single number between -1.0 (extremely negative) and 1.0 (extremely positive). "
            "Neutral sentiment should be 0.0. Do not include any other text, explanation, or context in your response."
        )
        response = engine.generate(system_prompt, text, max_tokens=10, temperature=0.0)
        
        # Try to extract a float from the response
        try:
            # strip out anything that isn't a number, period, or minus sign
            match = re.search(r'-?\d+\.?\d*', response)
            if match:
                score = float(match.group())
                # Clamp to [-1.0, 1.0] just in case
                return max(-1.0, min(1.0, score))
        except Exception:
            pass
            
        return 0.0 # fallback if LLM fails

class ReliabilityLayer:
    @staticmethod
    def check_consistency(response: str) -> bool:
        """Basic consistency check verifying length and non-empty status."""
        if not response or len(response.strip()) < 10:
            return False
        return True
        
    @staticmethod
    def check_hallucination(response: str, expected_context: str = "") -> bool:
        """Placeholder for checking basic hallucinations against expected context."""
        return True
