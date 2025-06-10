from abc import ABC, abstractmethod
from typing import Dict, Any, List
from src.domain.models.result import EvaluationMetric, MetricCalculationResult

class IMetricCalculator(ABC):
    """Abstract Base Class for metric calculation services."""

    @abstractmethod
    def supports_metric(self, metric: EvaluationMetric) -> bool:
        """
        Checks if this calculator supports the given evaluation metric.
        """
        pass

    @abstractmethod
    def calculate(
        self,
        llm_response: str,
        expected_result: str,
        metric: EvaluationMetric,
        details: Dict[str, Any] = None
    ) -> MetricCalculationResult:
        """
        Calculates the specified metric between the LLM's response and the expected result.
        """
        pass