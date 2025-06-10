from typing import List, Dict, Any
from src.domain.models.evaluation import EvaluationData, InputData, ExpectedResult
from src.domain.models.result import EvaluationMetric, MetricCalculationResult
from src.domain.models.task import AITaskName, AITaskStatus
from src.ports.llm_port import ILLMService
from src.ports.metrics_port import IMetricCalculator
from src.domain.services.domain_services import DataSetDomainService # Used if you were calculating aggregates

class EvaluateSingleDataUseCase:
    """
    Application use case for evaluating a single piece of data.
    This simplifies the flow by not requiring dataset management for the PoC.
    """
    def __init__(self,
                 llm_service: ILLMService,
                 metric_calculators: List[IMetricCalculator]):
        self._llm_service = llm_service
        # Map metric names (string values) to their supporting calculators for quick lookup
        self._metric_calculators: Dict[EvaluationMetric, IMetricCalculator] = {}
        for mc in metric_calculators:
            for m in EvaluationMetric:
                if mc.supports_metric(m):
                    self._metric_calculators[m] = mc

    def execute(self,
                ai_task_name: AITaskName,
                evaluation_metrics: List[EvaluationMetric],
                input_data: InputData,
                expected_result: ExpectedResult) -> EvaluationData:
        """
        Orchestrates the evaluation of a single data item.
        """
        evaluation_item = EvaluationData(
            input_data=input_data,
            expected_result=expected_result,
            status=AITaskStatus.IN_PROGRESS
        )

        # 1. Get LLM Response
        prompt = f"Task: {ai_task_name.value}\nInput: {input_data.decoded_data}"
        llm_response = self._llm_service.get_llm_response(prompt)
        evaluation_item.record_llm_response(llm_response)

        # 2. Calculate Metrics
        for metric in evaluation_metrics:
            if metric not in self._metric_calculators:
                # Log a warning or raise an exception if a requested metric has no calculator
                print(f"Warning: No calculator found for metric: {metric.value}")
                continue
            
            calculator = self._metric_calculators[metric]
            metric_result = calculator.calculate(
                llm_response=llm_response,
                expected_result=expected_result.decoded_result,
                metric=metric
            )
            evaluation_item.add_metric_result(metric_result.metric_name, metric_result.score, metric_result.details)
        
        evaluation_item.set_status(AITaskStatus.COMPLETED)
        return evaluation_item