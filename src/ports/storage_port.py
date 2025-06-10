from abc import ABC, abstractmethod
from typing import Optional, List
from src.domain.models.evaluation import EvaluationDataSet

class IEvaluationDataSetRepository(ABC):
    """Abstract Base Class for EvaluationDataSet repository."""

    @abstractmethod
    def save(self, dataset: EvaluationDataSet) -> EvaluationDataSet:
        """Saves or updates an evaluation dataset."""
        pass

    @abstractmethod
    def get_by_id(self, dataset_id: str) -> Optional[EvaluationDataSet]:
        """Retrieves an evaluation dataset by its ID."""
        pass

    @abstractmethod
    def get_all(self) -> List[EvaluationDataSet]:
        """Retrieves all evaluation datasets."""
        pass

    @abstractmethod
    def delete(self, dataset_id: str) -> None:
        """Deletes an evaluation dataset by its ID."""
        pass