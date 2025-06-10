# Enhanced LLM Evaluation Framework - Hexagonal Architecture with DDD
# Supporting Summarization, Translation, and Classification Tasks

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Any, Union, Generic, TypeVar
from enum import Enum
from datetime import datetime
import uuid
import json
import base64
from pathlib import Path
import asyncio

# =============================================================================
# DOMAIN LAYER - Core Business Logic
# =============================================================================

# Domain Value Objects
@dataclass(frozen=True)
class EvaluationId:
    value: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass(frozen=True)
class ModelId:
    value: str

@dataclass(frozen=True)
class DatasetId:
    value: str

@dataclass(frozen=True)
class SubDomainId:
    value: str

# Domain Enums
class EvaluationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskFamily(Enum):
    TEXT_GENERATION = "text_generation"
    TEXT_CLASSIFICATION = "text_classification"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"

class TaskType(Enum):
    # Text Generation Tasks
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    
    # Classification Tasks
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TOPIC_CLASSIFICATION = "topic_classification"
    INTENT_CLASSIFICATION = "intent_classification"

class InputDataType(Enum):
    TEXT = "text"
    STRUCTURED = "structured"
    MULTIMODAL = "multimodal"

class Theme(Enum):
    ACCURACY = "accuracy"
    FLUENCY = "fluency"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    SAFETY = "safety"
    BIAS = "bias"

# Domain Entities and Value Objects
@dataclass
class AITask:
    task_family: TaskFamily
    task_type: TaskType
    eval_metrics: List[str]
    method_type: str

    def get_applicable_metrics(self) -> List[str]:
        """Return metrics applicable for this task type"""
        return self.eval_metrics

@dataclass
class MetricsData:
    ai_task: AITask
    ai_task_metrics: List['AITaskMetrics']

    def get_metrics_info(self, task_name: str) -> Optional['AITaskMetrics']:
        for metric in self.ai_task_metrics:
            if metric.task_name == task_name:
                return metric
        return None

    def load_metrics_data(self, task_name: str) -> Dict[str, Any]:
        metric_info = self.get_metrics_info(task_name)
        if metric_info:
            return metric_info.to_dict()
        return {}

@dataclass
class AITaskMetrics:
    task_name: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "metrics": self.metrics,
            "metadata": self.metadata
        }

# Generic Input/Output Types
T = TypeVar('T')

@dataclass
class EvaluationInput(Generic[T]):
    data: T
    reference: Optional[T] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationOutput(Generic[T]):
    generated: T
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None

# Specific Input/Output Types
@dataclass
class SummarizationInput:
    text: str
    reference_summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TranslationInput:
    source_text: str
    source_language: str
    target_language: str
    reference_translation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ClassificationInput:
    text: str
    reference_label: Optional[str] = None
    possible_labels: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SummarizationOutput:
    generated_summary: str
    model_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TranslationOutput:
    translated_text: str
    detected_language: Optional[str] = None
    confidence: Optional[float] = None
    model_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ClassificationOutput:
    predicted_label: str
    confidence: float
    all_scores: Optional[Dict[str, float]] = None
    model_metadata: Dict[str, Any] = field(default_factory=dict)

# Evaluation Data Structure
@dataclass
class EvaluationData:
    input_data: Any
    input_data_type: InputDataType
    expected_result: Any

    def compute_metric(self, generated_output: Any, metric_name: str) -> float:
        """Compute a specific metric comparing expected vs generated output"""
        # This is a simplified implementation - in practice, you'd have
        # specific metric computation logic for each metric type
        if metric_name == "exact_match":
            return 1.0 if str(self.expected_result) == str(generated_output) else 0.0
        elif metric_name == "similarity":
            # Placeholder for similarity computation
            return 0.8  # Mock value
        return 0.0

# Enhanced Dataset Structure
@dataclass
class EvaluationDataSet:
    sub_domain_id: SubDomainId
    ai_task: AITask
    task_desc: Theme
    dataset: List[EvaluationData]
    theme: Theme
    metadata: Dict[str, Any]
    version: int

    def perform_evaluation(self, model_adapter: 'ModelAdapter') -> List['EvaluationMetric']:
        """Perform evaluation using the model adapter"""
        results = []
        for data in self.dataset:
            # This would be implemented based on task type
            pass
        return results

    def check_evaluation_status(self) -> EvaluationStatus:
        """Check the current status of evaluation"""
        return EvaluationStatus.PENDING

# Synthetic Data Generation
@dataclass
class SyntheticDataSet:
    sub_domain_type: str
    ai_task: AITask
    task_desc: Theme
    dataset: List[Any]
    theme: Theme
    metadata: Dict[str, Any]
    version: int

    def generate_synthetic_data(self, count: int = 100) -> List[Any]:
        """Generate synthetic data for evaluation"""
        synthetic_data = []
        for i in range(count):
            # Generate based on task type
            if self.ai_task.task_type == TaskType.SUMMARIZATION:
                synthetic_data.append({
                    "text": f"Sample text {i} for summarization testing. " * 10,
                    "reference_summary": f"Summary {i}"
                })
            elif self.ai_task.task_type == TaskType.TRANSLATION:
                synthetic_data.append({
                    "source_text": f"Hello world {i}",
                    "source_language": "en",
                    "target_language": "es",
                    "reference_translation": f"Hola mundo {i}"
                })
            elif self.ai_task.task_type == TaskType.SENTIMENT_ANALYSIS:
                synthetic_data.append({
                    "text": f"This is a {'positive' if i % 2 == 0 else 'negative'} review {i}",
                    "reference_label": "positive" if i % 2 == 0 else "negative"
                })
        return synthetic_data

    def validate_synthetic_data(self) -> bool:
        """Validate the quality of synthetic data"""
        return len(self.dataset) > 0

# Domain Data Structure for Managing Evaluations
@dataclass
class EvaluationDomainData:
    parent_domain_name: str
    subdomain_names: List[str]

    def add_sub_domain(self, parent: str, child: str) -> None:
        """Add a subdomain to a parent domain"""
        if parent == self.parent_domain_name and child not in self.subdomain_names:
            self.subdomain_names.append(child)

    def remove_sub_domain(self, parent: str, child: str) -> None:
        """Remove a subdomain from a parent domain"""
        if parent == self.parent_domain_name and child in self.subdomain_names:
            self.subdomain_names.remove(child)

    def get_domain_details(self, domain_id: str) -> str:
        """Get details about a specific domain"""
        if domain_id == self.parent_domain_name:
            return f"Parent domain: {self.parent_domain_name}, Subdomains: {len(self.subdomain_names)}"
        elif domain_id in self.subdomain_names:
            return f"Subdomain: {domain_id}, Parent: {self.parent_domain_name}"
        return "Domain not found"

# Central Evaluations Manager
@dataclass
class CentralEvaluations:
    dataset: List[EvaluationDataSet]
    metadata: List[Dict[str, Any]]
    version: int

    def perform_evaluation_all(self) -> List['EvaluationResult']:
        """Perform evaluation on all datasets"""
        results = []
        for ds in self.dataset:
            # Perform evaluation for each dataset
            pass
        return results

    def render_views(self) -> Dict[str, Any]:
        """Render evaluation results in different views"""
        return {
            "summary": f"Total datasets: {len(self.dataset)}",
            "by_task": self._group_by_task(),
            "by_status": self._group_by_status()
        }

    def render_leader_board(self) -> List[Dict[str, Any]]:
        """Render leaderboard of model performances"""
        return [
            {"model": "gpt-4", "score": 0.95, "task": "summarization"},
            {"model": "gpt-3.5", "score": 0.87, "task": "summarization"},
        ]

    def _group_by_task(self) -> Dict[str, int]:
        task_counts = {}
        for ds in self.dataset:
            task_name = ds.ai_task.task_type.value
            task_counts[task_name] = task_counts.get(task_name, 0) + 1
        return task_counts

    def _group_by_status(self) -> Dict[str, int]:
        status_counts = {}
        for ds in self.dataset:
            status = ds.check_evaluation_status().value
            status_counts[status] = status_counts.get(status, 0) + 1
        return status_counts

# Enhanced Evaluation Metric
@dataclass
class EvaluationMetric:
    name: str
    value: float
    task_type: TaskType
    theme: Theme
    metadata: Dict[str, Any] = field(default_factory=dict)

# Enhanced Evaluation Result
@dataclass
class EvaluationResult:
    evaluation_id: EvaluationId
    model_id: ModelId
    dataset_id: DatasetId
    sub_domain_id: SubDomainId
    task_type: TaskType
    metrics: List[EvaluationMetric]
    status: EvaluationStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

# Domain Services
class GeneralizedEvaluationService:
    def __init__(self, metric_calculators: Dict[TaskType, List['MetricCalculator']]):
        self._metric_calculators = metric_calculators

    def evaluate_task(
        self,
        task_type: TaskType,
        inputs: List[Any],
        outputs: List[Any]
    ) -> List[EvaluationMetric]:
        if task_type not in self._metric_calculators:
            raise ValueError(f"No metric calculators available for task type: {task_type}")

        calculators = self._metric_calculators[task_type]
        all_metrics = []

        for calculator in calculators:
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

class UnsupportedTaskTypeError(EvaluationDomainError):
    pass

# =============================================================================
# APPLICATION LAYER - Use Cases and Application Services
# =============================================================================

class EnhancedEvaluationOrchestrator:
    def __init__(
        self,
        model_adapter: 'EnhancedModelAdapter',
        dataset_adapter: 'EnhancedDatasetAdapter',
        evaluation_repository: 'EvaluationRepository',
        notification_adapter: 'NotificationAdapter',
        evaluation_service: GeneralizedEvaluationService,
        synthetic_data_generator: 'SyntheticDataGenerator'
    ):
        self._model_adapter = model_adapter
        self._dataset_adapter = dataset_adapter
        self._evaluation_repository = evaluation_repository
        self._notification_adapter = notification_adapter
        self._evaluation_service = evaluation_service
        self._synthetic_data_generator = synthetic_data_generator

    async def run_evaluation(
        self,
        model_id: ModelId,
        dataset_id: DatasetId,
        task_type: TaskType,
        sub_domain_id: Optional[SubDomainId] = None
    ) -> EvaluationResult:
        evaluation_id = EvaluationId()

        try:
            result = EvaluationResult(
                evaluation_id=evaluation_id,
                model_id=model_id,
                dataset_id=dataset_id,
                sub_domain_id=sub_domain_id or SubDomainId("default"),
                task_type=task_type,
                metrics=[],
                status=EvaluationStatus.PENDING,
                created_at=datetime.now()
            )

            await self._evaluation_repository.save(result)

            result.status = EvaluationStatus.RUNNING
            await self._evaluation_repository.save(result)

            # Load dataset based on task type
            dataset = await self._dataset_adapter.load_dataset(dataset_id, task_type)

            # Generate outputs using model
            outputs = []
            for input_data in dataset:
                output = await self._model_adapter.generate_output(
                    model_id, input_data, task_type
                )
                outputs.append(output)

            # Calculate metrics
            metrics = self._evaluation_service.evaluate_task(task_type, dataset, outputs)

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

    async def generate_synthetic_evaluation(
        self,
        model_id: ModelId,
        task_type: TaskType,
        count: int = 100
    ) -> EvaluationResult:
        """Generate synthetic data and run evaluation"""
        synthetic_dataset = await self._synthetic_data_generator.generate_dataset(
            task_type, count
        )

        # Create temporary dataset
        dataset_id = DatasetId(f"synthetic_{task_type.value}_{datetime.now().isoformat()}")
        await self._dataset_adapter.save_dataset(dataset_id, synthetic_dataset, task_type)

        return await self.run_evaluation(model_id, dataset_id, task_type)

# =============================================================================
# PORTS (INTERFACES) - Enhanced for Multiple Task Types
# =============================================================================

class EnhancedModelAdapter(Protocol):
    async def generate_output(
        self, model_id: ModelId, input_data: Any, task_type: TaskType
    ) -> Any:
        """Generate output based on task type"""
        ...

    async def generate_summary(
        self, model_id: ModelId, input_data: SummarizationInput
    ) -> SummarizationOutput:
        ...

    async def translate_text(
        self, model_id: ModelId, input_data: TranslationInput
    ) -> TranslationOutput:
        ...

    async def classify_text(
        self, model_id: ModelId, input_data: ClassificationInput
    ) -> ClassificationOutput:
        ...

    async def get_model_info(self, model_id: ModelId) -> Dict[str, Any]:
        ...

class EnhancedDatasetAdapter(Protocol):
    async def load_dataset(
        self, dataset_id: DatasetId, task_type: TaskType
    ) -> List[Any]:
        ...

    async def save_dataset(
        self, dataset_id: DatasetId, data: List[Any], task_type: TaskType
    ) -> None:
        ...

    async def get_dataset_info(self, dataset_id: DatasetId) -> Dict[str, Any]:
        ...

class SyntheticDataGenerator(Protocol):
    async def generate_dataset(
        self, task_type: TaskType, count: int
    ) -> List[Any]:
        ...

class MetricCalculator(Protocol):
    def calculate(self, inputs: List[Any], outputs: List[Any]) -> EvaluationMetric:
        ...

# =============================================================================
# ADAPTERS (IMPLEMENTATIONS) - Enhanced for Multiple Tasks
# =============================================================================

class OpenAIEnhancedModelAdapter:
    def __init__(self, api_key: str):
        self._api_key = api_key

    async def generate_output(
        self, model_id: ModelId, input_data: Any, task_type: TaskType
    ) -> Any:
        """Route to appropriate generation method based on task type"""
        if task_type == TaskType.SUMMARIZATION:
            return await self.generate_summary(model_id, input_data)
        elif task_type == TaskType.TRANSLATION:
            return await self.translate_text(model_id, input_data)
        elif task_type in [TaskType.SENTIMENT_ANALYSIS, TaskType.TOPIC_CLASSIFICATION, TaskType.INTENT_CLASSIFICATION]:
            return await self.classify_text(model_id, input_data)
        else:
            raise UnsupportedTaskTypeError(f"Task type {task_type} not supported")

    async def generate_summary(
        self, model_id: ModelId, input_data: SummarizationInput
    ) -> SummarizationOutput:
        await self._simulate_api_delay()

        # Mock OpenAI API call for summarization
        prompt = f"Summarize the following text:\n\n{input_data.text}"
        summary = f"Summary: {input_data.text[:100]}..." if len(input_data.text) > 100 else f"Summary: {input_data.text}"

        return SummarizationOutput(
            generated_summary=summary,
            model_metadata={
                "model": model_id.value,
                "tokens_used": len(input_data.text.split()),
                "prompt_tokens": len(prompt.split())
            }
        )

    async def translate_text(
        self, model_id: ModelId, input_data: TranslationInput
    ) -> TranslationOutput:
        await self._simulate_api_delay()

        # Mock translation
        translations = {
            ("en", "es"): {"Hello": "Hola", "world": "mundo", "How are you": "Cómo estás"},
            ("en", "fr"): {"Hello": "Bonjour", "world": "monde", "How are you": "Comment allez-vous"},
            ("es", "en"): {"Hola": "Hello", "mundo": "world", "Cómo estás": "How are you"}
        }

        lang_pair = (input_data.source_language, input_data.target_language)
        if lang_pair in translations:
            translated = input_data.source_text
            for source, target in translations[lang_pair].items():
                translated = translated.replace(source, target)
        else:
            translated = f"[Translated to {input_data.target_language}] {input_data.source_text}"

        return TranslationOutput(
            translated_text=translated,
            detected_language=input_data.source_language,
            confidence=0.95,
            model_metadata={
                "model": model_id.value,
                "source_lang": input_data.source_language,
                "target_lang": input_data.target_language
            }
        )

    async def classify_text(
        self, model_id: ModelId, input_data: ClassificationInput
    ) -> ClassificationOutput:
        await self._simulate_api_delay()

        # Mock classification based on simple heuristics
        text_lower = input_data.text.lower()

        if input_data.possible_labels:
            # Use provided labels
            if "positive" in input_data.possible_labels and any(
                word in text_lower for word in ["good", "great", "excellent", "positive", "love"]
            ):
                predicted = "positive"
                confidence = 0.9
            elif "negative" in input_data.possible_labels and any(
                word in text_lower for word in ["bad", "terrible", "awful", "negative", "hate"]
            ):
                predicted = "negative"
                confidence = 0.9
            else:
                predicted = input_data.possible_labels[0]  # Default to first label
                confidence = 0.6
        else:
            # Default sentiment analysis
            if any(word in text_lower for word in ["good", "great", "excellent", "love"]):
                predicted = "positive"
                confidence = 0.85
            elif any(word in text_lower for word in ["bad", "terrible", "awful", "hate"]):
                predicted = "negative"
                confidence = 0.85
            else:
                predicted = "neutral"
                confidence = 0.7

        return ClassificationOutput(
            predicted_label=predicted,
            confidence=confidence,
            all_scores={predicted: confidence, "other": 1 - confidence},
            model_metadata={
                "model": model_id.value,
                "classification_method": "heuristic"
            }
        )

    async def get_model_info(self, model_id: ModelId) -> Dict[str, Any]:
        return {
            "model_id": model_id.value,
            "provider": "openai",
            "supported_tasks": [task.value for task in TaskType],
            "max_tokens": 4096
        }

    async def _simulate_api_delay(self):
        await asyncio.sleep(0.1)

class EnhancedJSONDatasetAdapter:
    def __init__(self, data_directory: Path):
        self._data_directory = data_directory

    async def load_dataset(
        self, dataset_id: DatasetId, task_type: TaskType
    ) -> List[Any]:
        file_path = self._data_directory / f"{dataset_id.value}_{task_type.value}.json"

        if not file_path.exists():
            # Try without task type suffix
            file_path = self._data_directory / f"{dataset_id.value}.json"

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return self._convert_to_input_objects(data, task_type)

    async def save_dataset(
        self, dataset_id: DatasetId, data: List[Any], task_type: TaskType
    ) -> None:
        file_path = self._data_directory / f"{dataset_id.value}_{task_type.value}.json"

        # Convert objects to serializable format
        serializable_data = []
        for item in data:
            if hasattr(item, '__dict__'):
                serializable_data.append(item.__dict__)
            else:
                serializable_data.append(item)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)

    def _convert_to_input_objects(self, data: List[Dict], task_type: TaskType) -> List[Any]:
        """Convert raw data to appropriate input objects based on task type"""
        if task_type == TaskType.SUMMARIZATION:
            return [
                SummarizationInput(
                    text=item['text'],
                    reference_summary=item.get('reference_summary'),
                    metadata=item.get('metadata', {})
                ) for item in data
            ]
        elif task_type == TaskType.TRANSLATION:
            return [
                TranslationInput(
                    source_text=item['source_text'],
                    source_language=item['source_language'],
                    target_language=item['target_language'],
                    reference_translation=item.get('reference_translation'),
                    metadata=item.get('metadata', {})
                ) for item in data
            ]
        elif task_type in [TaskType.SENTIMENT_ANALYSIS, TaskType.TOPIC_CLASSIFICATION, TaskType.INTENT_CLASSIFICATION]:
            return [
                ClassificationInput(
                    text=item['text'],
                    reference_label=item.get('reference_label'),
                    possible_labels=item.get('possible_labels'),
                    metadata=item.get('metadata', {})
                ) for item in data
            ]
        else:
            return data

    async def get_dataset_info(self, dataset_id: DatasetId) -> Dict[str, Any]:
        # Try to find any file with this dataset_id
        pattern = f"{dataset_id.value}*.json"
        matching_files = list(self._data_directory.glob(pattern))

        if not matching_files:
            return {"error": "Dataset not found"}

        info = {
            "dataset_id": dataset_id.value,
            "files": [f.name for f in matching_files],
            "tasks_available": []
        }

        for file_path in matching_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    info["size"] = len(data)
                    # Infer task type from filename or data structure
                    if "translation" in file_path.name:
                        info["tasks_available"].append("translation")
                    elif "classification" in file_path.name or "sentiment" in file_path.name:
                        info["tasks_available"].append("classification")
                    elif "summarization" in file_path.name:
                        info["tasks_available"].append("summarization")
                    break
            except:
                continue

        return info

class DefaultSyntheticDataGenerator:
    async def generate_dataset(
        self, task_type: TaskType, count: int
    ) -> List[Any]:
        """Generate synthetic data based on task type"""
        if task_type == TaskType.SUMMARIZATION:
            return self._generate_summarization_data(count)
        elif task_type == TaskType.TRANSLATION:
            return self._generate_translation_data(count)
        elif task_type in [TaskType.SENTIMENT_ANALYSIS, TaskType.TOPIC_CLASSIFICATION]:
            return self._generate_classification_data(task_type, count)
        else:
            raise UnsupportedTaskTypeError(f"Synthetic data generation not supported for {task_type}")

    def _generate_summarization_data(self, count: int) -> List[SummarizationInput]:
        data = []
        templates = [
            "The recent study on climate change reveals significant impacts on global weather patterns. Researchers have found that...",
            "In the field of artificial intelligence, machine learning algorithms are being developed to solve complex problems...",
            "The economic analysis shows trends in market behavior that indicate potential future changes in consumer spending..."
        ]

        for i in range(count):
            template = templates[i % len(templates)]
            full_text = f"{template} {' '.join(['Additional context'] * 20)} Sample {i}."
            summary = f"{template.split('.')[0]}."

            data.append(SummarizationInput(
                text=full_text,
                reference_summary=summary,
                metadata={"generated": True, "template_id": i % len(templates)}
            ))

        return data

    def _generate_translation_data(self, count: int) -> List[TranslationInput]:
        data = []
        phrases = [
            ("Hello, how are you?", "Hola, ¿cómo estás?"),
            ("The weather is nice today.", "El clima está agradable hoy."),
            ("I love learning new languages.", "Me encanta aprender nuevos idiomas.")
        ]

        for i in range(count):
            english, spanish = phrases[i % len(phrases)]

            data.append(TranslationInput(
                source_text=f"{english} Sample {i}.",
                source_language="en",
                target_language="es",
                reference_translation=f"{spanish} Muestra {i}.",
                metadata={"generated": True, "phrase_id": i % len(phrases)}
            ))

        return data

    def _generate_classification_data(self, task_type: TaskType, count: int) -> List[ClassificationInput]:
        data = []

        if task_type == TaskType.SENTIMENT_ANALYSIS:
            positive_texts = [
                "I love this product! It's amazing and works perfectly.",
                "Excellent service and great quality. Highly recommended!",
                "This is fantastic! Exceeded my expectations completely."
            ]
            negative_texts = [
                "Terrible product. Doesn't work as advertised.",
                "Awful service and poor quality. Very disappointed.",
                "This is the worst purchase I've ever made."
            ]

            for i in range(count):
                if i % 2 == 0:
                    text = positive_texts[i % len(positive_texts)]
                    label = "positive"
                else:
                    text = negative_texts[i % len(negative_texts)]
                    label = "negative"

                data.append(ClassificationInput(
                    text=f"{text} Sample {i}.",
                    reference_label=label,
                    possible_labels=["positive", "negative", "neutral"],
                    metadata={"generated": True, "sentiment": label}
                ))

        elif task_type == TaskType.TOPIC_CLASSIFICATION:
            topics = {
                "technology": ["AI and machine learning are revolutionizing the industry.",
                             "The latest smartphone features incredible camera technology."],
                "sports": ["The football match was exciting with a last-minute goal.",
                          "Tennis championship shows amazing athletic performance."],
                "politics": ["The recent policy changes affect economic growth.",
                           "Election results show significant shifts in voter preference."]
            }

            topic_list = list(topics.keys())
            for i in range(count):
                topic = topic_list[i % len(topic_list)]
                text = topics[topic][i % len(topics[topic])]

                data.append(ClassificationInput(
                    text=f"{text} Sample {i}.",
                    reference_label=topic,
                    possible_labels=topic_list,
                    metadata={"generated": True, "topic": topic}
                ))

        return data

# Enhanced Metric Calculators for Different Tasks
class ROUGECalculator:
    def calculate(self, inputs: List[SummarizationInput], outputs: List[SummarizationOutput]) -> EvaluationMetric:
        scores = []
        for inp, out in zip(inputs, outputs):
            if inp.reference_summary:
                ref_words = set(inp.reference_summary.lower().split())
                gen_words = set(out.generated_summary.lower().split())
                if ref_words:
                    score = len(ref_words.intersection(gen_words)) / len(ref_words)
                    scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return EvaluationMetric(
            name="ROUGE-L",
            value=avg_score,
            task_type=TaskType.SUMMARIZATION,
            theme=Theme.ACCURACY,
            metadata={"samples_evaluated": len(scores)}
        )

class BLEUCalculator:
    def calculate(self, inputs: List[TranslationInput], outputs: List[TranslationOutput]) -> EvaluationMetric:
        scores = []
        for inp, out in zip(inputs, outputs):
            if inp.reference_translation:
                # Simplified BLEU calculation (n-gram overlap)
                ref_words = inp.reference_translation.lower().split()
                gen_words = out.translated_text.lower().split()
                
                if ref_words and gen_words:
                    # 1-gram precision
                    ref_1grams = set(ref_words)
                    gen_1grams = set(gen_words)
                    precision = len(ref_1grams.intersection(gen_1grams)) / len(gen_1grams) if gen_1grams else 0
                    scores.append(precision)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return EvaluationMetric(
            name="BLEU",
            value=avg_score,
            task_type=TaskType.TRANSLATION,
            theme=Theme.ACCURACY,
            metadata={"samples_evaluated": len(scores)}
        )

class AccuracyCalculator:
    def calculate(self, inputs: List[ClassificationInput], outputs: List[ClassificationOutput]) -> EvaluationMetric:
        correct = 0
        total = 0
        
        for inp, out in zip(inputs, outputs):
            if inp.reference_label:
                total += 1
                if inp.reference_label == out.predicted_label:
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return EvaluationMetric(
            name="Accuracy",
            value=accuracy,
            task_type=TaskType.SENTIMENT_ANALYSIS,  # Can be used for all classification tasks
            theme=Theme.ACCURACY,
            metadata={"correct": correct, "total": total}
        )

class F1ScoreCalculator:
    def calculate(self, inputs: List[ClassificationInput], outputs: List[ClassificationOutput]) -> EvaluationMetric:
        # Calculate F1 score for binary/multi-class classification
        label_stats = {}
        
        for inp, out in zip(inputs, outputs):
            if inp.reference_label:
                true_label = inp.reference_label
                pred_label = out.predicted_label
                
                # Initialize stats for labels
                for label in [true_label, pred_label]:
                    if label not in label_stats:
                        label_stats[label] = {"tp": 0, "fp": 0, "fn": 0}
                
                # Update stats
                if true_label == pred_label:
                    label_stats[true_label]["tp"] += 1
                else:
                    label_stats[pred_label]["fp"] += 1
                    label_stats[true_label]["fn"] += 1

        # Calculate macro F1
        f1_scores = []
        for label, stats in label_stats.items():
            precision = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0
            recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)

        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        return EvaluationMetric(
            name="F1-Score",
            value=macro_f1,
            task_type=TaskType.SENTIMENT_ANALYSIS,
            theme=Theme.ACCURACY,
            metadata={"macro_f1": macro_f1, "per_label_stats": label_stats}
        )

# Enhanced Repository with better query capabilities
class EnhancedInMemoryEvaluationRepository:
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

    async def find_by_task_type(self, task_type: TaskType) -> List[EvaluationResult]:
        return [
            result for result in self._storage.values()
            if result.task_type == task_type
        ]

    async def find_by_status(self, status: EvaluationStatus) -> List[EvaluationResult]:
        return [
            result for result in self._storage.values()
            if result.status == status
        ]

    async def get_leaderboard(self, task_type: TaskType, metric_name: str) -> List[Dict[str, Any]]:
        """Get leaderboard for a specific task and metric"""
        results = await self.find_by_task_type(task_type)
        completed_results = [r for r in results if r.status == EvaluationStatus.COMPLETED]
        
        leaderboard = []
        for result in completed_results:
            for metric in result.metrics:
                if metric.name == metric_name:
                    leaderboard.append({
                        "model_id": result.model_id.value,
                        "dataset_id": result.dataset_id.value,
                        "metric_value": metric.value,
                        "evaluation_id": result.evaluation_id.value,
                        "completed_at": result.completed_at
                    })
                    break
        
        # Sort by metric value (descending)
        leaderboard.sort(key=lambda x: x["metric_value"], reverse=True)
        return leaderboard

# Enhanced API Controller
class EnhancedFastAPIEvaluationController:
    def __init__(self, orchestrator: EnhancedEvaluationOrchestrator):
        self._orchestrator = orchestrator

    async def start_evaluation(
        self, model_id: str, dataset_id: str, task_type: str, sub_domain_id: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            task_enum = TaskType(task_type)
            sub_domain = SubDomainId(sub_domain_id) if sub_domain_id else None
            
            result = await self._orchestrator.run_evaluation(
                ModelId(model_id),
                DatasetId(dataset_id),
                task_enum,
                sub_domain
            )
            
            return {
                "evaluation_id": result.evaluation_id.value,
                "status": result.status.value,
                "task_type": result.task_type.value,
                "created_at": result.created_at.isoformat()
            }
        except Exception as e:
            return {"error": str(e), "status": "failed"}

    async def start_synthetic_evaluation(
        self, model_id: str, task_type: str, count: int = 100
    ) -> Dict[str, Any]:
        try:
            task_enum = TaskType(task_type)
            result = await self._orchestrator.generate_synthetic_evaluation(
                ModelId(model_id), task_enum, count
            )
            
            return {
                "evaluation_id": result.evaluation_id.value,
                "status": result.status.value,
                "task_type": result.task_type.value,
                "synthetic_samples": count,
                "created_at": result.created_at.isoformat()
            }
        except Exception as e:
            return {"error": str(e), "status": "failed"}

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
            "sub_domain_id": result.sub_domain_id.value,
            "task_type": result.task_type.value,
            "status": result.status.value,
            "metrics": [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "task_type": metric.task_type.value,
                    "theme": metric.theme.value,
                    "metadata": metric.metadata
                } for metric in result.metrics
            ],
            "created_at": result.created_at.isoformat(),
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
            "error_message": result.error_message
        }

    async def get_leaderboard(self, task_type: str, metric_name: str) -> Dict[str, Any]:
        try:
            task_enum = TaskType(task_type)
            leaderboard = await self._orchestrator._evaluation_repository.get_leaderboard(
                task_enum, metric_name
            )
            
            return {
                "task_type": task_type,
                "metric_name": metric_name,
                "leaderboard": leaderboard,
                "total_entries": len(leaderboard)
            }
        except Exception as e:
            return {"error": str(e)}

    async def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get overall evaluation summary across all tasks"""
        try:
            repo = self._orchestrator._evaluation_repository
            
            summary = {
                "total_evaluations": len(repo._storage),
                "by_status": {},
                "by_task_type": {},
                "recent_evaluations": []
            }
            
            # Count by status
            for result in repo._storage.values():
                status = result.status.value
                summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
                
                task_type = result.task_type.value
                summary["by_task_type"][task_type] = summary["by_task_type"].get(task_type, 0) + 1
            
            # Get recent evaluations
            recent = sorted(
                repo._storage.values(),
                key=lambda x: x.created_at,
                reverse=True
            )[:10]
            
            summary["recent_evaluations"] = [
                {
                    "evaluation_id": r.evaluation_id.value,
                    "model_id": r.model_id.value,
                    "task_type": r.task_type.value,
                    "status": r.status.value,
                    "created_at": r.created_at.isoformat()
                } for r in recent
            ]
            
            return summary
        except Exception as e:
            return {"error": str(e)}

# Enhanced Factory with support for all task types
class EnhancedEvaluationFrameworkFactory:
    @staticmethod
    def create_evaluation_system(
        openai_api_key: str,
        data_directory: Path
    ) -> tuple[EnhancedEvaluationOrchestrator, EnhancedFastAPIEvaluationController]:
        
        # Create adapters
        model_adapter = OpenAIEnhancedModelAdapter(openai_api_key)
        dataset_adapter = EnhancedJSONDatasetAdapter(data_directory)
        evaluation_repository = EnhancedInMemoryEvaluationRepository()
        notification_adapter = ConsoleNotificationAdapter()
        synthetic_data_generator = DefaultSyntheticDataGenerator()

        # Create metric calculators for different task types
        metric_calculators = {
            TaskType.SUMMARIZATION: [
                ROUGECalculator(),
                BERTScoreCalculator()  # Reuse for semantic similarity
            ],
            TaskType.TRANSLATION: [
                BLEUCalculator(),
                BERTScoreCalculator()  # For semantic similarity
            ],
            TaskType.SENTIMENT_ANALYSIS: [
                AccuracyCalculator(),
                F1ScoreCalculator()
            ],
            TaskType.TOPIC_CLASSIFICATION: [
                AccuracyCalculator(),
                F1ScoreCalculator()
            ],
            TaskType.INTENT_CLASSIFICATION: [
                AccuracyCalculator(),
                F1ScoreCalculator()
            ]
        }

        # Create domain service
        evaluation_service = GeneralizedEvaluationService(metric_calculators)

        # Create orchestrator
        orchestrator = EnhancedEvaluationOrchestrator(
            model_adapter=model_adapter,
            dataset_adapter=dataset_adapter,
            evaluation_repository=evaluation_repository,
            notification_adapter=notification_adapter,
            evaluation_service=evaluation_service,
            synthetic_data_generator=synthetic_data_generator
        )

        # Create controller
        controller = EnhancedFastAPIEvaluationController(orchestrator)

        return orchestrator, controller

# =============================================================================
# COMPREHENSIVE USAGE EXAMPLE
# =============================================================================

async def comprehensive_demo():
    """Comprehensive demo showing all supported task types"""
    
    # Setup
    data_dir = Path("./datasets")
    data_dir.mkdir(exist_ok=True)

    # Create sample datasets for different tasks
    
    # 1. Summarization Dataset
    summarization_data = [
        {
            "text": "Climate change is causing significant impacts on global weather patterns. Rising temperatures are leading to more frequent extreme weather events, including hurricanes, droughts, and flooding. Scientists warn that immediate action is needed to reduce greenhouse gas emissions and mitigate these effects.",
            "reference_summary": "Climate change causes extreme weather; immediate action needed to reduce emissions.",
            "metadata": {"source": "climate_report", "difficulty": "medium"}
        },
        {
            "text": "Artificial intelligence and machine learning technologies are revolutionizing various industries. From healthcare to finance, AI systems are helping automate complex tasks, improve decision-making, and enhance user experiences. However, ethical considerations around AI deployment remain a key concern.",
            "reference_summary": "AI is transforming industries but raises ethical concerns.",
            "metadata": {"source": "tech_news", "difficulty": "easy"}
        }
    ]

    # 2. Translation Dataset
    translation_data = [
        {
            "source_text": "Good morning, how are you today?",
            "source_language": "en",
            "target_language": "es",
            "reference_translation": "Buenos días, ¿cómo estás hoy?",
            "metadata": {"difficulty": "easy", "domain": "greeting"}
        },
        {
            "source_text": "The artificial intelligence conference will discuss the latest research developments.",
            "source_language": "en",
            "target_language": "fr",
            "reference_translation": "La conférence sur l'intelligence artificielle discutera des derniers développements de recherche.",
            "metadata": {"difficulty": "hard", "domain": "technical"}
        }
    ]

    # 3. Classification Dataset (Sentiment Analysis)
    sentiment_data = [
        {
            "text": "I absolutely love this product! It exceeded all my expectations and works perfectly.",
            "reference_label": "positive",
            "possible_labels": ["positive", "negative", "neutral"],
            "metadata": {"confidence": "high", "domain": "product_review"}
        },
        {
            "text": "This service is terrible. I'm very disappointed with the quality and support.",
            "reference_label": "negative",
            "possible_labels": ["positive", "negative", "neutral"],
            "metadata": {"confidence": "high", "domain": "service_review"}
        },
        {
            "text": "The weather today is okay, nothing special but not bad either.",
            "reference_label": "neutral",
            "possible_labels": ["positive", "negative", "neutral"],
            "metadata": {"confidence": "medium", "domain": "general"}
        }
    ]

    # Save datasets
    datasets = [
        ("summarization_test", summarization_data),
        ("translation_test", translation_data),
        ("sentiment_test", sentiment_data)
    ]

    for dataset_name, data in datasets:
        with open(data_dir / f"{dataset_name}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # Create enhanced system
    orchestrator, controller = EnhancedEvaluationFrameworkFactory.create_evaluation_system(
        openai_api_key="fake-openai-key-for-demo",
        data_directory=data_dir
    )

    print("🚀 Starting Comprehensive LLM Evaluation Demo")
    print("=" * 60)

    # Test different task types
    tasks_to_test = [
        ("gpt-4", "summarization_test", TaskType.SUMMARIZATION),
        ("gpt-4", "translation_test", TaskType.TRANSLATION),
        ("gpt-3.5-turbo", "sentiment_test", TaskType.SENTIMENT_ANALYSIS)
    ]

    evaluation_ids = []

    # 1. Run standard evaluations
    print("\n📊 Running Standard Evaluations...")
    for model_id, dataset_id, task_type in tasks_to_test:
        print(f"\n🔄 Starting {task_type.value} evaluation...")
        result = await controller.start_evaluation(model_id, dataset_id, task_type.value)
        
        if "evaluation_id" in result:
            evaluation_ids.append(result["evaluation_id"])
            print(f"✅ Evaluation started: {result['evaluation_id']}")
        else:
            print(f"❌ Failed to start evaluation: {result.get('error', 'Unknown error')}")

    # Wait for evaluations to complete
    await asyncio.sleep(2)

    # 2. Display results
    print(f"\n📈 Evaluation Results:")
    print("-" * 40)
    
    for eval_id in evaluation_ids:
        result = await controller.get_evaluation_results(eval_id)
        if "error" not in result:
            print(f"\n🎯 Evaluation: {eval_id[:8]}...")
            print(f"   Model: {result['model_id']}")
            print(f"   Task: {result['task_type']}")
            print(f"   Status: {result['status']}")
            print(f"   Metrics:")
            for metric in result['metrics']:
                print(f"     • {metric['name']}: {metric['value']:.3f}")

    # 3. Run synthetic data evaluations
    print(f"\n🧪 Running Synthetic Data Evaluations...")
    synthetic_tasks = [TaskType.SUMMARIZATION, TaskType.TRANSLATION, TaskType.SENTIMENT_ANALYSIS]
    
    for task_type in synthetic_tasks:
        print(f"\n🔄 Generating synthetic {task_type.value} evaluation...")
        result = await controller.start_synthetic_evaluation("gpt-3.5-turbo", task_type.value, 50)
        
        if "evaluation_id" in result:
            print(f"✅ Synthetic evaluation started: {result['evaluation_id'][:8]}...")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")

    # Wait for synthetic evaluations
    await asyncio.sleep(2)

    # 4. Display leaderboards
    print(f"\n🏆 Leaderboards:")
    print("-" * 40)
    
    leaderboard_queries = [
        (TaskType.SUMMARIZATION, "ROUGE-L"),
        (TaskType.TRANSLATION, "BLEU"),
        (TaskType.SENTIMENT_ANALYSIS, "Accuracy")
    ]
    
    for task_type, metric_name in leaderboard_queries:
        leaderboard = await controller.get_leaderboard(task_type.value, metric_name)
        
        if "error" not in leaderboard:
            print(f"\n🥇 {task_type.value.title()} - {metric_name}:")
            for i, entry in enumerate(leaderboard["leaderboard"][:3], 1):
                print(f"   {i}. {entry['model_id']}: {entry['metric_value']:.3f}")

    # 5. Display overall summary
    print(f"\n📊 Overall Summary:")
    print("-" * 40)
    
    summary = await controller.get_evaluation_summary()
    if "error" not in summary:
        print(f"Total Evaluations: {summary['total_evaluations']}")
        print(f"By Status: {summary['by_status']}")
        print(f"By Task Type: {summary['by_task_type']}")
        
        print(f"\nRecent Evaluations:")
        for eval_info in summary['recent_evaluations'][:5]:
            print(f"  • {eval_info['evaluation_id'][:8]}... - {eval_info['task_type']} ({eval_info['status']})")

    print(f"\n🎉 Demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(comprehensive_demo())