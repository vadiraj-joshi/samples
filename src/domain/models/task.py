# src/llm_evaluation/domain/models/task.py
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass, field
import uuid

class EvaluationMetric(str, Enum):
    ACCURACY = "Accuracy"
    PRECISION = "Precision"
    RECALL = "Recall"
    F1_SCORE = "F1-Score"
    BLEU = "BLEU"
    ROUGE = "ROUGE"
    BERT_SCORE = "BERTScore" # Added as per new structure

class AITaskName(str, Enum):
    SUMMARIZATION = "Summarization"
    TRANSLATION = "Translation"
    YoutubeING = "Question Answering"
    CODE_GENERATION = "Code Generation"
    TEXT_CLASSIFICATION = "Text Classification"

class AITaskStatus(str, Enum):
    """Enumeration for the status of an AI task evaluation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class AITaskType(str, Enum):
    GENERATIVE = "Generative"
    DISCRIMINATIVE = "Discriminative"

class EvaluationDataType(str, Enum):
    TEXT = "Text"
    IMAGE = "Image"
    AUDIO = "Audio"

class Theme(str, Enum):
    GENERAL = "General"
    LEGAL = "Legal"
    MEDICAL = "Medical"
    TECHNICAL = "Technical"
    FINANCE = "Finance"

class EvaluationStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

@dataclass
class AITask:
    """
    Domain Entity representing an AI Task definition.
    """
    task_id: str
    task_family: AITaskType
    task_name: AITaskName
    evaluation_metrics: List['EvaluationMetric'] = field(default_factory=list) # Forward ref
    method_type: Optional[str] = None # e.g., 'generative', 'discriminative'

    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())