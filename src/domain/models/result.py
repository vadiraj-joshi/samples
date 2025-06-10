from enum import Enum
from typing import Dict, Any
from pydantic import BaseModel, Field

class EvaluationMetric(str, Enum):
    """Enumeration for different evaluation metrics."""
    # Summarization specific
    SUMMARIZATION_ACCURACY = "summarization_accuracy"
    ROUGE = "rouge"

    # Translation specific
    TRANSLATION_FLUENCY = "translation_fluency"
    BLEU = "bleu"

    # General
    ACCURACY = "accuracy"

class EvaluationDataType(str, Enum):
    """Enumeration for types of data (e.g., text, image, audio)."""
    TEXT = "text"
    # IMAGE = "image"
    # AUDIO = "audio"

class MetricCalculationResult(BaseModel):
    """Represents the result of a single metric calculation."""
    metric_name: EvaluationMetric
    score: float = Field(..., ge=0.0, le=1.0) # Score typically between 0 and 1
    details: Dict[str, Any] = Field(default_factory=dict) # Optional detailed results

class EvaluationResult(BaseModel):
    """Represents the results for a single evaluation data item across all metrics."""
    llm_response: str
    metric_results: Dict[EvaluationMetric, MetricCalculationResult] = Field(default_factory=dict)
    # status: EvaluationStatus # Can be added for detailed tracking per item