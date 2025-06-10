# LLM Evaluation Framework - Hexagonal Architecture with Domain Driven Design
# Example: Text Summarization Evaluation

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Any
from enum import Enum
from datetime import datetime
import uuid
import json
from pathlib import Path

# =============================================================================
# DOMAIN LAYER - Core Business Logic
# =============================================================================

# Domain Entities
@dataclass(frozen=True)
class EvaluationId:
    value: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass(frozen=True)
class ModelId:
    value: str

@dataclass(frozen=True)
class DatasetId:
    value: str

class EvaluationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class SummarizationInput:
    text: str
    reference_summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SummarizationOutput:
    generated_summary: str
    model_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationMetric:
    name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationResult:
    evaluation_id: EvaluationId
    model_id: ModelId
    dataset_id: DatasetId
    metrics: List[EvaluationMetric]
    status: EvaluationStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

# Domain Services
class SummarizationEvaluationService:
    def __init__(self, metric_calculators: List['MetricCalculator']):
        self._metric_calculators = metric_calculators
    
    def evaluate_summarization(
        self, 
        inputs: List[SummarizationInput], 
        outputs: List[SummarizationOutput]
    ) -> List[EvaluationMetric]:
        if len(inputs) != len(outputs):
            raise ValueError("Input and output lists must have the same length")
        
        all_metrics = []
        for calculator in self._metric_calculators:
            metric = calculator.calculate(inputs, outputs)
            all_metrics.append(metric)
        
        return all_metrics

# Domain Exceptions
class EvaluationDomainError(Exception):
    pass

class InvalidEvaluationStateError(EvaluationDomainError):
    pass

class MetricCalculationError(EvaluationDomainError):
    pass

# =============================================================================
# APPLICATION LAYER - Use Cases and Application Services
# =============================================================================

# Application Services
class EvaluationOrchestrator:
    def __init__(
        self,
        model_adapter: 'ModelAdapter',
        dataset_adapter: 'DatasetAdapter',
        evaluation_repository: 'EvaluationRepository',
        notification_adapter: 'NotificationAdapter',
        evaluation_service: SummarizationEvaluationService
    ):
        self._model_adapter = model_adapter
        self._dataset_adapter = dataset_adapter
        self._evaluation_repository = evaluation_repository
        self._notification_adapter = notification_adapter
        self._evaluation_service = evaluation_service
    
    async def run_evaluation(
        self, 
        model_id: ModelId, 
        dataset_id: DatasetId
    ) -> EvaluationResult:
        evaluation_id = EvaluationId()
        
        try:
            # Create initial evaluation result
            result = EvaluationResult(
                evaluation_id=evaluation_id,
                model_id=model_id,
                dataset_id=dataset_id,
                metrics=[],
                status=EvaluationStatus.PENDING,
                created_at=datetime.now()
            )
            
            await self._evaluation_repository.save(result)
            
            # Update status to running
            result.status = EvaluationStatus.RUNNING
            await self._evaluation_repository.save(result)
            
            # Load dataset
            dataset = await self._dataset_adapter.load_dataset(dataset_id)
            
            # Generate summaries using model
            outputs = []
            for input_data in dataset:
                output = await self._model_adapter.generate_summary(model_id, input_data)
                outputs.append(output)
            
            # Calculate metrics
            metrics = self._evaluation_service.evaluate_summarization(dataset, outputs)
            
            # Update result with metrics
            result.metrics = metrics
            result.status = EvaluationStatus.COMPLETED
            result.completed_at = datetime.now()
            
            await self._evaluation_repository.save(result)
            await self._notification_adapter.notify_completion(result)
            
            return result
            
        except Exception as e:
            result.status = EvaluationStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.now()
            
            await self._evaluation_repository.save(result)
            await self._notification_adapter.notify_failure(result)
            
            raise EvaluationDomainError(f"Evaluation failed: {str(e)}")

# =============================================================================
# PORTS (INTERFACES) - Hexagonal Architecture Boundaries
# =============================================================================

# Primary Ports (Driving/Left Side)
class EvaluationAPI(Protocol):
    async def start_evaluation(self, model_id: str, dataset_id: str) -> Dict[str, Any]:
        ...
    
    async def get_evaluation_status(self, evaluation_id: str) -> Dict[str, Any]:
        ...
    
    async def get_evaluation_results(self, evaluation_id: str) -> Dict[str, Any]:
        ...

# Secondary Ports (Driven/Right Side)
class ModelAdapter(Protocol):
    async def generate_summary(self, model_id: ModelId, input_data: SummarizationInput) -> SummarizationOutput:
        ...
    
    async def get_model_info(self, model_id: ModelId) -> Dict[str, Any]:
        ...

class DatasetAdapter(Protocol):
    async def load_dataset(self, dataset_id: DatasetId) -> List[SummarizationInput]:
        ...
    
    async def get_dataset_info(self, dataset_id: DatasetId) -> Dict[str, Any]:
        ...

class EvaluationRepository(Protocol):
    async def save(self, result: EvaluationResult) -> None:
        ...
    
    async def find_by_id(self, evaluation_id: EvaluationId) -> Optional[EvaluationResult]:
        ...
    
    async def find_by_model_and_dataset(self, model_id: ModelId, dataset_id: DatasetId) -> List[EvaluationResult]:
        ...

class NotificationAdapter(Protocol):
    async def notify_completion(self, result: EvaluationResult) -> None:
        ...
    
    async def notify_failure(self, result: EvaluationResult) -> None:
        ...

class MetricCalculator(Protocol):
    def calculate(self, inputs: List[SummarizationInput], outputs: List[SummarizationOutput]) -> EvaluationMetric:
        ...

# =============================================================================
# ADAPTERS (IMPLEMENTATIONS) - Infrastructure Layer
# =============================================================================

# Primary Adapters (Web/API Controllers)
class FastAPIEvaluationController:
    def __init__(self, orchestrator: EvaluationOrchestrator):
        self._orchestrator = orchestrator
    
    async def start_evaluation(self, model_id: str, dataset_id: str) -> Dict[str, Any]:
        try:
            result = await self._orchestrator.run_evaluation(
                ModelId(model_id), 
                DatasetId(dataset_id)
            )
            return {
                "evaluation_id": result.evaluation_id.value,
                "status": result.status.value,
                "created_at": result.created_at.isoformat()
            }
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    async def get_evaluation_status(self, evaluation_id: str) -> Dict[str, Any]:
        result = await self._orchestrator._evaluation_repository.find_by_id(
            EvaluationId(evaluation_id)
        )
        
        if not result:
            return {"error": "Evaluation not found"}
        
        return {
            "evaluation_id": result.evaluation_id.value,
            "status": result.status.value,
            "created_at": result.created_at.isoformat(),
            "completed_at": result.completed_at.isoformat() if result.completed_at else None
        }
    
    async def get_evaluation_results(self, evaluation_id: str) -> Dict[str, Any]:
        result = await self._orchestrator._evaluation_repository.find_by_id(
            EvaluationId(evaluation_id)
        )
        
        if not result:
            return {"error": "Evaluation not found"}
        
        return {
            "evaluation_id": result.evaluation_id.value,
            "model_id": result.model_id.value,
            "dataset_id": result.dataset_id.value,
            "status": result.status.value,
            "metrics": [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "metadata": metric.metadata
                } for metric in result.metrics
            ],
            "created_at": result.created_at.isoformat(),
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
            "error_message": result.error_message
        }

# Secondary Adapters (Infrastructure Implementations)
class OpenAIModelAdapter:
    def __init__(self, api_key: str):
        self._api_key = api_key
    
    async def generate_summary(self, model_id: ModelId, input_data: SummarizationInput) -> SummarizationOutput:
        # Simulate OpenAI API call
        # In real implementation, you would use openai library
        await self._simulate_api_delay()
        
        # Mock response
        summary = f"Summary of: {input_data.text[:50]}..." if len(input_data.text) > 50 else f"Summary of: {input_data.text}"
        
        return SummarizationOutput(
            generated_summary=summary,
            model_metadata={"model": model_id.value, "tokens_used": len(input_data.text.split())}
        )
    
    async def get_model_info(self, model_id: ModelId) -> Dict[str, Any]:
        return {"model_id": model_id.value, "provider": "openai", "type": "text-generation"}
    
    async def _simulate_api_delay(self):
        import asyncio
        await asyncio.sleep(0.1)  # Simulate network delay

class JSONFileDatasetAdapter:
    def __init__(self, data_directory: Path):
        self._data_directory = data_directory
    
    async def load_dataset(self, dataset_id: DatasetId) -> List[SummarizationInput]:
        file_path = self._data_directory / f"{dataset_id.value}.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return [
            SummarizationInput(
                text=item['text'],
                reference_summary=item.get('reference_summary'),
                metadata=item.get('metadata', {})
            ) for item in data
        ]
    
    async def get_dataset_info(self, dataset_id: DatasetId) -> Dict[str, Any]:
        dataset = await self.load_dataset(dataset_id)
        return {
            "dataset_id": dataset_id.value,
            "size": len(dataset),
            "has_references": any(item.reference_summary for item in dataset)
        }

class InMemoryEvaluationRepository:
    def __init__(self):
        self._storage: Dict[str, EvaluationResult] = {}
    
    async def save(self, result: EvaluationResult) -> None:
        self._storage[result.evaluation_id.value] = result
    
    async def find_by_id(self, evaluation_id: EvaluationId) -> Optional[EvaluationResult]:
        return self._storage.get(evaluation_id.value)
    
    async def find_by_model_and_dataset(self, model_id: ModelId, dataset_id: DatasetId) -> List[EvaluationResult]:
        return [
            result for result in self._storage.values()
            if result.model_id == model_id and result.dataset_id == dataset_id
        ]

class ConsoleNotificationAdapter:
    async def notify_completion(self, result: EvaluationResult) -> None:
        print(f"✅ Evaluation {result.evaluation_id.value} completed successfully!")
        print(f"   Model: {result.model_id.value}")
        print(f"   Dataset: {result.dataset_id.value}")
        print(f"   Metrics: {len(result.metrics)} calculated")
    
    async def notify_failure(self, result: EvaluationResult) -> None:
        print(f"❌ Evaluation {result.evaluation_id.value} failed!")
        print(f"   Error: {result.error_message}")

# Metric Calculator Implementations
class ROUGECalculator:
    def calculate(self, inputs: List[SummarizationInput], outputs: List[SummarizationOutput]) -> EvaluationMetric:
        # Simplified ROUGE calculation (in real implementation, use rouge-score library)
        scores = []
        for inp, out in zip(inputs, outputs):
            if inp.reference_summary:
                # Simple word overlap as proxy for ROUGE
                ref_words = set(inp.reference_summary.lower().split())
                gen_words = set(out.generated_summary.lower().split())
                if ref_words:
                    score = len(ref_words.intersection(gen_words)) / len(ref_words)
                    scores.append(score)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return EvaluationMetric(
            name="ROUGE-L",
            value=avg_score,
            metadata={"samples_evaluated": len(scores)}
        )

class BERTScoreCalculator:
    def calculate(self, inputs: List[SummarizationInput], outputs: List[SummarizationOutput]) -> EvaluationMetric:
        # Simplified BERTScore calculation (in real implementation, use bert-score library)
        scores = []
        for inp, out in zip(inputs, outputs):
            if inp.reference_summary:
                # Simple length ratio as proxy for semantic similarity
                ref_len = len(inp.reference_summary.split())
                gen_len = len(out.generated_summary.split())
                score = min(ref_len, gen_len) / max(ref_len, gen_len) if max(ref_len, gen_len) > 0 else 0
                scores.append(score)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return EvaluationMetric(
            name="BERTScore",
            value=avg_score,
            metadata={"samples_evaluated": len(scores)}
        )

# =============================================================================
# DEPENDENCY INJECTION & COMPOSITION ROOT
# =============================================================================

class EvaluationFrameworkFactory:
    @staticmethod
    def create_evaluation_system(
        openai_api_key: str,
        data_directory: Path
    ) -> tuple[EvaluationOrchestrator, FastAPIEvaluationController]:
        # Create adapters
        model_adapter = OpenAIModelAdapter(openai_api_key)
        dataset_adapter = JSONFileDatasetAdapter(data_directory)
        evaluation_repository = InMemoryEvaluationRepository()
        notification_adapter = ConsoleNotificationAdapter()
        
        # Create metric calculators
        metric_calculators = [
            ROUGECalculator(),
            BERTScoreCalculator()
        ]
        
        # Create domain service
        evaluation_service = SummarizationEvaluationService(metric_calculators)
        
        # Create orchestrator
        orchestrator = EvaluationOrchestrator(
            model_adapter=model_adapter,
            dataset_adapter=dataset_adapter,
            evaluation_repository=evaluation_repository,
            notification_adapter=notification_adapter,
            evaluation_service=evaluation_service
        )
        
        # Create controller
        controller = FastAPIEvaluationController(orchestrator)
        
        return orchestrator, controller

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

async def main():
    # Setup
    data_dir = Path("./datasets")
    data_dir.mkdir(exist_ok=True)
    
    # Create sample dataset
    sample_data = [
        {
            "text": "The quick brown fox jumps over the lazy dog. This is a sample text for summarization testing.",
            "reference_summary": "A fox jumps over a dog.",
            "metadata": {"source": "test"}
        },
        {
            "text": "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
            "reference_summary": "ML is part of AI focusing on algorithms.",
            "metadata": {"source": "tech"}
        }
    ]
    
    with open(data_dir / "test_dataset.json", "w") as f:
        json.dump(sample_data, f, indent=2)
    
    # Create system
    orchestrator, controller = EvaluationFrameworkFactory.create_evaluation_system(
        openai_api_key="fake-key-for-demo",
        data_directory=data_dir
    )
    
    # Run evaluation
    print("Starting evaluation...")
    result = await controller.start_evaluation("gpt-3.5-turbo", "test_dataset")
    print(f"Evaluation started: {result}")
    
    # Check results
    if "evaluation_id" in result:
        import asyncio
        await asyncio.sleep(1)  # Wait for completion
        
        final_result = await controller.get_evaluation_results(result["evaluation_id"])
        print(f"Final results: {json.dumps(final_result, indent=2)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
