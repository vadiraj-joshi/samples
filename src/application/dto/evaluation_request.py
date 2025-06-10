from pydantic import BaseModel
from typing import List, Optional
from src.domain.models.task import AITaskName
from src.domain.models.result import EvaluationMetric
from src.domain.models.evaluation import InputData, ExpectedResult

class EvaluateSingleDataRequest(BaseModel):
    """Request DTO for evaluating a single piece of data."""
    ai_task_name: AITaskName
    evaluation_metrics: List[EvaluationMetric]
    input_data: InputData
    expected_result: ExpectedResult