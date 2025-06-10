# src/llm_evaluation/ports/data_provision_port.py
from abc import ABC, abstractmethod
from typing import List

from src.domain.models.evaluation import EvaluationData
from src.domain.models.task import AITaskName


class ISyntheticDataSetGenerator(ABC):
    """Driven Port: Interface for generating synthetic evaluation datasets."""
    @abstractmethod
    def generate_synthetic_data(
        self,
        ai_task_name: AITaskName,
        num_samples: int = 1
    ) -> List[EvaluationData]:
        """Generates a list of synthetic EvaluationData samples."""
        pass

    @abstractmethod
    def validate_synthetic_data(self, dataset: List[EvaluationData]) -> bool:
        """Validates the structure and content of synthetic data."""
        pass