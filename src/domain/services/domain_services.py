from typing import List, Dict
from src.domain.models.evaluation import EvaluationDataSet, EvaluationData
from src.domain.models.result import EvaluationMetric, EvaluationResult
from statistics import mean

class DataSetDomainService:
    """
    Domain service for performing business logic operations related to EvaluationDataSet,
    such as calculating aggregated scores.
    """
    def calculate_overall_scores(self, dataset: EvaluationDataSet) -> Dict[EvaluationMetric, float]:
        """
        Calculates the average score for each metric across all completed evaluation data items
        in a dataset.
        """
        overall_scores: Dict[EvaluationMetric, List[float]] = {metric: [] for metric in EvaluationMetric}

        for item in dataset.evaluation_data:
            if item.evaluation_result and item.evaluation_result.metric_results:
                for metric_name, metric_res in item.evaluation_result.metric_results.items():
                    overall_scores[metric_name].append(metric_res.score)
        
        # Calculate average for each metric
        averaged_scores: Dict[EvaluationMetric, float] = {}
        for metric, scores_list in overall_scores.items():
            if scores_list:
                averaged_scores[metric] = mean(scores_list)
            else:
                averaged_scores[metric] = 0.0 # Or NaN, depending on preference for no data

        return averaged_scores