from typing import Dict, List, Optional
from src.domain.models.evaluation import EvaluationDataSet
from src.ports.storage_port import IEvaluationDataSetRepository

# In-memory store (for PoC purposes)
_DATASET_STORE: Dict[str, EvaluationDataSet] = {}

class InMemoryEvaluationDataSetRepository(IEvaluationDataSetRepository):
    """
    In-memory implementation of IEvaluationDataSetRepository for PoC.
    """
    def save(self, dataset: EvaluationDataSet) -> EvaluationDataSet:
        _DATASET_STORE[dataset.dataset_id] = dataset
        return dataset

    def get_by_id(self, dataset_id: str) -> Optional[EvaluationDataSet]:
        return _DATASET_STORE.get(dataset_id)

    def get_all(self) -> List[EvaluationDataSet]:
        return list(_DATASET_STORE.values())

    def delete(self, dataset_id: str) -> None:
        _DATASET_STORE.pop(dataset_id, None)

    def clear_all(self): # Helper for testing/dev
        _DATASET_STORE.clear()