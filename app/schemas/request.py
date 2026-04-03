# app/schemas/request.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class SeverityLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class ComplaintRequest(BaseModel):
    """Input schema for citizen complaint"""
    complaint_text: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Raw complaint text from citizen",
        examples=["Loud music party next door every night, unbearable"]
    )
    borough: Optional[str] = Field(
        None,
        description="NYC borough (optional)",
        examples=["MANHATTAN", "BROOKLYN"]
    )
    zip_code: Optional[str] = Field(
        None,
        description="ZIP code (optional)",
        examples=["10001"]
    )

    @field_validator("complaint_text")
    @classmethod
    def validate_complaint_text(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Complaint text must be at least 3 characters")
        return v


class DepartmentPrediction(BaseModel):
    """Department classification result"""
    department: str = Field(..., description="Predicted department")
    confidence: float = Field(..., description="Prediction confidence (0-1)")


class SentimentPrediction(BaseModel):
    """Sentiment analysis result"""
    sentiment: str = Field(..., description="Predicted sentiment class")
    confidence: float = Field(..., description="Prediction confidence (0-1)")


class PriorityScoring(BaseModel):
    """Priority score details"""
    priority_score: float = Field(..., description="Priority score (0-4)")
    severity: SeverityLevel = Field(..., description="Severity level")
    recommended_action: str = Field(..., description="Recommended action based on priority")


class ComplaintResponse(BaseModel):
    """Full API response"""
    status: str = Field("success", description="Response status")
    input_text: str = Field(..., description="Original input text")
    cleaned_text: str = Field(..., description="Preprocessed text")
    department: DepartmentPrediction
    sentiment: SentimentPrediction
    priority: PriorityScoring
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    models_loaded: dict
    version: str = "1.0.0"