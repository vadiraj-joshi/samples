import random
from typing import Dict, Any
from src.ports.metrics_port import IMetricCalculator, MetricCalculationResult
from src.domain.models.result import EvaluationMetric

class SummarizationMetricAdapter(IMetricCalculator):
    """
    A dummy metric calculator for summarization tasks.
    It returns a random score for PoC purposes.
    """
    def supports_metric(self, metric: EvaluationMetric) -> bool:
        return metric in [EvaluationMetric.SUMMARIZATION_ACCURACY, EvaluationMetric.ROUGE]

    def calculate(
        self,
        llm_response: str,
        expected_result: str,
        metric: EvaluationMetric,
        details: Dict[str, Any] = None
    ) -> MetricCalculationResult:
        if not self.supports_metric(metric):
            raise ValueError(f"Metric {metric.value} not supported by SummarizationMetricAdapter.")

        # Simulate a score based on random for demo
        score = random.uniform(0.6, 0.95) # High scores for demo optimism!

        if metric == EvaluationMetric.SUMMARIZATION_ACCURACY:
            score = round(random.uniform(0.7, 0.99), 2)
            details = {"method": "dummy_string_overlap"}
        elif metric == EvaluationMetric.ROUGE:
            score = round(random.uniform(0.65, 0.90), 2)
            details = {"rouge_l": score, "rouge_1": round(random.uniform(0.6,0.8),2)}
        
        return MetricCalculationResult(
            metric_name=metric,
            score=score,
            details=details or {}
        )