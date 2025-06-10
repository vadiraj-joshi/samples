# src/llm_evaluation/ports/evaluation_port.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from src.domain.models.evaluation import EvaluationDomainData, EvaluationDataSet
from src.domain.models.task import AITaskName, EvaluationDataType, Theme
from src.domain.models.result import EvaluationMetric

class IEvaluationService(ABC):
    """
    Driving Port: Interface for the main application evaluation capabilities.
    This defines what the "outside world" (e.g., REST API) can ask the application to do.
    """
    @abstractmethod
    async def create_evaluation_domain(
        self, parent_domain_name: str, description: Optional[str] = None
    ) -> EvaluationDomainData:
        pass

    @abstractmethod
    def get_evaluation_domain(self, domain_id: str) -> EvaluationDomainData:
        pass

    @abstractmethod
    def get_all_evaluation_domains(self) -> List[EvaluationDomainData]:
        pass

    @abstractmethod
    async def create_evaluation_dataset(
        self,
        sub_domain_id: str,
        ai_task_name: AITaskName,
        task_description: str,
        theme: Theme,
        metadata: Dict[str, Any] = None,
        version: int = 1
    ) -> EvaluationDataSet:
        pass

    @abstractmethod
    async def add_evaluation_data_to_dataset(
        self,
        dataset_id: str,
        input_data_base64: str,
        input_data_type: EvaluationDataType,
        expected_result_base64: str
    ) -> EvaluationDataSet:
        pass

    @abstractmethod
    async def generate_and_add_synthetic_data(
        self, dataset_id: str, num_samples: int = 1
    ) -> EvaluationDataSet:
        pass

    @abstractmethod
    async def perform_evaluation_on_dataset(
        self, dataset_id: str, evaluation_metric: EvaluationMetric
    ) -> EvaluationDataSet:
        pass

    @abstractmethod
    def get_evaluation_dataset(self, dataset_id: str) -> EvaluationDataSet:
        pass

    @abstractmethod
    def get_all_evaluation_datasets(self) -> List[EvaluationDataSet]:
        pass

    @abstractmethod
    def get_dataset_evaluation_status(self, dataset_id: str) -> Dict[str, int]:
        pass

    @abstractmethod
    def get_leaderboard_data(self, ai_task_name: Optional[AITaskName] = None) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def register_ai_task_metrics(
        self,
        ai_task_name: AITaskName,
        available_metrics: List[EvaluationMetric],
        description: Optional[str] = None
    ) -> None:
        pass

    @abstractmethod
    def get_available_metrics_for_task(self, ai_task_name: AITaskName) -> Optional[List[EvaluationMetric]]:
        pass