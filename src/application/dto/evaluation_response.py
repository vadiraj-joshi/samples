from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from src.domain.models.task import AITaskName, AITaskStatus
from src.domain.models.result import EvaluationMetric, MetricCalculationResult, EvaluationResult
from src.domain.models.evaluation import InputData, ExpectedResult

class EvaluateSingleDataResponse(BaseModel):
    """Response DTO for a single data evaluation."""
    item_id: str
    ai_task_name: AITaskName
    input_data: InputData
    expected_result: ExpectedResult
    llm_response: str
    metric_results: Dict[EvaluationMetric, MetricCalculationResult]
    status: AITaskStatus