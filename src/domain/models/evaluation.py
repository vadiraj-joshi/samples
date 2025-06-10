from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
import base64

from src.domain.models.task import AITaskStatus, AITaskName
from src.domain.models.result import EvaluationResult, EvaluationDataType, EvaluationMetric, MetricCalculationResult

class InputData(BaseModel):
    """Represents the input provided to the LLM."""
    data_base64: str = Field(description="Base64 encoded input data (e.g., text, image).")
    data_type: EvaluationDataType = Field(description="Type of the input data (e.g., TEXT).")
    decoded_data: Optional[str] = Field(None, description="Decoded input data (for convenience, not persisted usually).")

    def __init__(self, **data):
        super().__init__(**data)
        if self.data_base64 and not self.decoded_data:
            self.decoded_data = base64.b64decode(self.data_base64).decode('utf-8')

class ExpectedResult(BaseModel):
    """Represents the expected output or ground truth for the LLM's response."""
    result_base64: str = Field(description="Base64 encoded expected result (ground truth).")
    decoded_result: Optional[str] = Field(None, description="Decoded expected result (for convenience, not persisted usually).")

    def __init__(self, **data):
        super().__init__(**data)
        if self.result_base64 and not self.decoded_result:
            self.decoded_result = base64.b64decode(self.result_base64).decode('utf-8')

class EvaluationData(BaseModel):
    """Represents a single piece of evaluation data (input, expected, LLM response, results)."""
    item_id: str = Field(default_factory=lambda: str(uuid4()))
    input_data: InputData
    expected_result: ExpectedResult
    evaluation_result: Optional[EvaluationResult] = None # Stores LLM response and metric results
    ai_task_name: AITaskName = AITaskName.SUMMARIZATION
    status: AITaskStatus = AITaskStatus.PENDING
    
    def record_llm_response(self, response: str):
        if self.evaluation_result:
            self.evaluation_result.llm_response = response
        else:
            self.evaluation_result = EvaluationResult(llm_response=response)
        self.status = AITaskStatus.IN_PROGRESS # Or a more granular status like LLM_COMPLETE

    def add_metric_result(self, metric_result: EvaluationMetric, score: float, details: Optional[Dict[str, Any]] = None):
        if not self.evaluation_result:
            self.evaluation_result = EvaluationResult(llm_response="") # Initialize if not present
        
        calculated_result = MetricCalculationResult(
            metric_name=metric_result,
            score=score,
            details=details or {}
        )
        self.evaluation_result.metric_results[metric_result] = calculated_result
        # Update status if all metrics are done or if this is the final step
        # For simplicity, we'll mark as completed after all metrics are processed in use case.
    def set_status(self, new_status: AITaskStatus):
        self.status = new_status

class EvaluationDataSet(BaseModel):
    """Represents a collection of evaluation data items."""
    dataset_id: str = Field(default_factory=lambda: str(uuid4()))
    dataset_name: str
    description: Optional[str] = None
    ai_task_name: AITaskName
    evaluation_data: List[EvaluationData] = Field(default_factory=list)
    status: AITaskStatus = AITaskStatus.PENDING # Overall dataset status

    def add_evaluation_data(self, data: EvaluationData):
        self.evaluation_data.append(data)

    def set_status(self, new_status: AITaskStatus):
        self.status = new_status