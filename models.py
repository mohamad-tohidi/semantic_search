"""
Simple Pydantic models for semantic search.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel


class FlexibleModel(BaseModel):
    """Allow extra fields and flexible types."""
    model_config = {"extra": "allow", "strict": False}


class QuestionMetaData(FlexibleModel):
    """Question metadata."""
    category: Optional[str] = None
    keywords: Optional[List[str]] = None
    source_name: Optional[str] = None
    url: Optional[str] = None


class Question(FlexibleModel):
    """Question with text and embeddings."""
    language: str = "fa"
    text: Dict[str, str]  # {"fa": "text", "en": "text"}
    e5_search_vector: Optional[List[float]] = None
    bge_search_vector: Optional[List[float]] = None
    metadata: Optional[QuestionMetaData] = None


class AnswerMetaData(FlexibleModel):
    """Answer metadata."""
    source_name: Optional[str] = None
    source_link: Optional[str] = None


class Answer(FlexibleModel):
    """Answer with text and embeddings."""
    language: str = "fa"
    text: Dict[str, str]  # {"fa": "text", "en": "text"}
    summary: Optional[Dict[str, str]] = None
    e5_search_vector: Optional[List[float]] = None
    bge_search_vector: Optional[List[float]] = None
    metadata: Optional[AnswerMetaData] = None


class MetaData(FlexibleModel):
    """Document metadata."""
    original_question: Optional[str] = None
    errors: Optional[List[Any]] = None
    viewer: Optional[int] = 0
    was_not_question: Optional[bool] = False
    title: Optional[str] = None


class QARecord(FlexibleModel):
    """Main QA record."""
    elastic_id: Union[str, int]
    question: Question
    answers: List[List[Answer]]
    metadata: MetaData
    processed_by: Optional[str] = None
    processed_at: Optional[str] = None

