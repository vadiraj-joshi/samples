from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated, List

# Application Layer
from src.application.use_cases.evaluate import EvaluateSingleDataUseCase
from src.application.dto.evaluation_request import EvaluateSingleDataRequest
from src.application.dto.evaluation_response import EvaluateSingleDataResponse

# Ports (interfaces)
from src.ports.llm_port import ILLMService
from src.ports.metrics_port import IMetricCalculator
from src.ports.storage_port import IEvaluationDataSetRepository # Not strictly used in this simplified router, but kept for context

# Adapters (concrete implementations)
from src.adapters.llm.openai_adapter import DummyLLMAdapter
from src.adapters.metrics.accuracy_metric_adapter import SummarizationMetricAdapter
from src.adapters.metrics.bleu_metric_adapter import TranslationMetricAdapter
from src.adapters.storage.inmemory_adapter import InMemoryEvaluationDataSetRepository # Not strictly used, but kept for context

# --- Dependency Injection Setup ---
# Instantiate your Adapters as singletons for the app's lifetime
dummy_llm_service = DummyLLMAdapter()
summarization_metric_calc = SummarizationMetricAdapter()
translation_metric_calc = TranslationMetricAdapter()

# You can add more metric calculators here if needed
all_metric_calculators: List[IMetricCalculator] = [
    summarization_metric_calc,
    translation_metric_calc
]

def get_evaluate_single_data_use_case() -> EvaluateSingleDataUseCase:
    """
    FastAPI dependency injector that provides a configured EvaluateSingleDataUseCase instance.
    """
    return EvaluateSingleDataUseCase(
        llm_service=dummy_llm_service,
        metric_calculators=all_metric_calculators
    )

# --- FastAPI Router Definition ---
router = APIRouter(prefix="/evaluation", tags=["Single Data Evaluation PoC"])

@router.post(
    "/single",
    response_model=EvaluateSingleDataResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate a single piece of LLM data for a specific task and metrics"
)
async def evaluate_single_data(
    request: EvaluateSingleDataRequest,
    use_case: Annotated[EvaluateSingleDataUseCase, Depends(get_evaluate_single_data_use_case)]
):
    try:
        evaluated_item = use_case.execute(
            ai_task_name=request.ai_task_name,
            evaluation_metrics=request.evaluation_metrics,
            input_data=request.input_data,
            expected_result=request.expected_result
        )
        return EvaluateSingleDataResponse(
            item_id=evaluated_item.item_id,
            ai_task_name=evaluated_item.ai_task_name, # Not strictly used in this PoC's domain model, but good for response DTO
            input_data=evaluated_item.input_data,
            expected_result=evaluated_item.expected_result,
            llm_response=evaluated_item.evaluation_result.llm_response,
            metric_results=evaluated_item.evaluation_result.metric_results,
            status=evaluated_item.status
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))