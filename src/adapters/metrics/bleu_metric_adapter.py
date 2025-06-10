import random
from typing import Dict, Any
from src.ports.metrics_port import IMetricCalculator, MetricCalculationResult
from src.domain.models.result import EvaluationMetric

class TranslationMetricAdapter(IMetricCalculator):
    """
    A dummy metric calculator for translation tasks.
    It returns a random score for PoC purposes.
    """
    def supports_metric(self, metric: EvaluationMetric) -> bool:
        return metric in [EvaluationMetric.TRANSLATION_FLUENCY, EvaluationMetric.BLEU]

    def calculate(
        self,
        llm_response: str,
        expected_result: str,
        metric: EvaluationMetric,
        details: Dict[str, Any] = None
    ) -> MetricCalculationResult:
        if not self.supports_metric(metric):
            raise ValueError(f"Metric {metric.value} not supported by TranslationMetricAdapter.")

        # Simulate a score
        score = random.uniform(0.5, 0.9)

        if metric == EvaluationMetric.TRANSLATION_FLUENCY:
            score = round(random.uniform(0.75, 0.98), 2)
            details = {"rating_model": "dummy_neural_net"}
        elif metric == EvaluationMetric.BLEU:
            score = round(random.uniform(0.6, 0.85), 2)
            details = {"bleu_n_gram": 4} # Simplified
        
        return MetricCalculationResult(
            metric_name=metric,
            score=score,
            details=details or {}
        )