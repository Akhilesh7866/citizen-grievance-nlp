# app/main.py
import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import ComplaintRequest, ComplaintResponse, HealthResponse, SeverityLevel
from .models import DepartmentClassifier, SentimentAnalyzer
from .utils import clean_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global model instances
department_classifier: Optional[DepartmentClassifier] = None
sentiment_analyzer: Optional[SentimentAnalyzer] = None

# Model paths (can be overridden via environment variables)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "artifacts")

RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_department_model.pkl")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
BERT_MODEL_PATH = os.path.join(MODEL_DIR, "bert_sentiment_model")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "sentiment_label_encoder.pkl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global department_classifier, sentiment_analyzer
    
    logger.info("Starting application...")
    logger.info(f"Model directory: {MODEL_DIR}")
    
    # Load Department Classifier
    try:
        department_classifier = DepartmentClassifier(RF_MODEL_PATH, TFIDF_PATH)
        logger.info("✓ Department classifier loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load department classifier: {e}")
        department_classifier = None
    
    # Load Sentiment Analyzer
    try:
        sentiment_analyzer = SentimentAnalyzer(BERT_MODEL_PATH, LABEL_ENCODER_PATH)
        logger.info("✓ Sentiment analyzer loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load sentiment analyzer: {e}")
        sentiment_analyzer = None
    
    yield
    
    # Cleanup
    logger.info("Shutting down application...")
    department_classifier = None
    sentiment_analyzer = None


# Create FastAPI app
app = FastAPI(
    title="AI-Driven Citizen Grievance Analysis API",
    description="""
    ## Overview
    This API analyzes citizen complaints and provides:
    - **Department Classification**: Routes complaints to appropriate civic departments
    - **Sentiment Analysis**: Classifies complaint urgency (Positive/Neutral/Negative/Critical)
    - **Priority Scoring**: Calculates mathematical urgency score for triage
    
    ## Models Used
    - **Random Forest** for department classification (Week 2)
    - **BERT (bert-base-uncased)** for sentiment analysis (Week 3)
    
    ## Priority Score Formula
    `priority_score = base_score × confidence`
    
    | Sentiment | Base Score | Severity |
    |-----------|------------|----------|
    | Critical/Urgent | 4.0 | Critical |
    | Negative | 3.0 | High |
    | Neutral | 2.0 | Medium |
    | Positive | 1.0 | Low |
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== ENDPOINTS ==============

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """Root endpoint - redirects to health check."""
    return await health_check()


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model loading status."""
    return HealthResponse(
        status="healthy" if department_classifier and sentiment_analyzer else "degraded",
        models_loaded={
            "department_classifier": department_classifier.is_loaded() if department_classifier else False,
            "sentiment_analyzer": sentiment_analyzer.is_loaded() if sentiment_analyzer else False,
        },
        version="1.0.0"
    )


@app.post("/predict", response_model=ComplaintResponse, tags=["Prediction"])
async def predict_complaint(request: ComplaintRequest):
    """
    Analyze a citizen complaint and return predictions.
    
    This endpoint:
    1. Preprocesses the input text
    2. Predicts the responsible department
    3. Analyzes sentiment and urgency
    4. Calculates priority score
    """
    # Check if models are loaded
    if not department_classifier or not sentiment_analyzer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please try again later."
        )
    
    start_time = time.time()
    
    try:
        # Step 1: Preprocess text
        cleaned_text = clean_text(request.complaint_text)
        
        if not cleaned_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Complaint text could not be processed. Please provide valid text."
            )
        
        # Step 2: Department Prediction
        department, dept_confidence = department_classifier.predict(cleaned_text)
        
        # Step 3: Sentiment & Priority Analysis
        priority_result = sentiment_analyzer.get_priority_score(cleaned_text)
        sentiment = priority_result["sentiment"]
        sent_confidence = priority_result["confidence"]
        
        # Step 4: Build response
        processing_time = round(time.time() - start_time, 4)
        
        response = ComplaintResponse(
            status="success",
            input_text=request.complaint_text,
            cleaned_text=cleaned_text,
            department={
                "department": department,
                "confidence": dept_confidence
            },
            sentiment={
                "sentiment": sentiment,
                "confidence": sent_confidence
            },
            priority={
                "priority_score": priority_result["priority_score"],
                "severity": priority_result["severity"],
                "recommended_action": priority_result["recommended_action"]
            },
            metadata={
                "processing_time_seconds": processing_time,
                "borough": request.borough,
                "zip_code": request.zip_code
            }
        )
        
        logger.info(f"Prediction completed in {processing_time}s - Dept: {department}, Sentiment: {sentiment}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(complaints: list[ComplaintRequest]):
    """
    Analyze multiple complaints in a single request.
    
    Returns a list of predictions in the same order as input.
    """
    if not department_classifier or not sentiment_analyzer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please try again later."
        )
    
    if len(complaints) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 100 complaints per batch request."
        )
    
    results = []
    for complaint in complaints:
        try:
            cleaned_text = clean_text(complaint.complaint_text)
            
            if cleaned_text:
                department, dept_conf = department_classifier.predict(cleaned_text)
                priority_result = sentiment_analyzer.get_priority_score(cleaned_text)
                
                results.append({
                    "status": "success",
                    "input_text": complaint.complaint_text,
                    "department": department,
                    "sentiment": priority_result["sentiment"],
                    "priority_score": priority_result["priority_score"],
                    "severity": priority_result["severity"]
                })
            else:
                results.append({
                    "status": "error",
                    "input_text": complaint.complaint_text,
                    "error": "Could not process text"
                })
        except Exception as e:
            results.append({
                "status": "error",
                "input_text": complaint.complaint_text,
                "error": str(e)
            })
    
    return {
        "total": len(complaints),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "results": results
    }


@app.get("/models/info", tags=["Models"])
async def get_model_info():
    """Get information about loaded models."""
    info = {
        "department_classifier": {
            "loaded": department_classifier.is_loaded() if department_classifier else False,
            "classes": list(department_classifier.classes) if department_classifier else [],
            "type": "Random Forest"
        },
        "sentiment_analyzer": {
            "loaded": sentiment_analyzer.is_loaded() if sentiment_analyzer else False,
            "classes": list(sentiment_analyzer.label_encoder.classes_) if sentiment_analyzer else [],
            "device": sentiment_analyzer.device if sentiment_analyzer else None,
            "type": "BERT (bert-base-uncased)"
        }
    }
    return info


# ============== ERROR HANDLERS ==============

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "detail": "An unexpected error occurred.",
            "error": str(exc)
        }
    )
