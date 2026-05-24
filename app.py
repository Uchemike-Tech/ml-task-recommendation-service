# ml-service/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import joblib
import numpy as np
import pandas as pd
import os

# Initialize FastAPI
app = FastAPI(title="Task Completion ML Service", 
              description="Predicts next task based on completed tasks",
              version="1.0.0")

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and metadata
MODEL_PATH = "best_task_model.joblib"
SCALER_PATH = "scaler.joblib"
METADATA_PATH = "complete_metadata.joblib"

# Global variables for loaded models
model = None
scaler = None
metadata = None
task_df = None
dependencies = None
task_names = None
category_names = None
n_tasks = None

@app.on_event("startup")
async def load_models():
    """Load all models and metadata on startup"""
    global model, scaler, metadata, task_df, dependencies, task_names, category_names, n_tasks
    
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        metadata = joblib.load(METADATA_PATH)
        
        task_df = pd.DataFrame(metadata['task_df'])
        dependencies = metadata['dependencies']
        task_names = metadata['task_names']
        category_names = metadata['category_names']
        n_tasks = metadata['n_tasks']
        
        print(f"✅ Model loaded successfully!")
        print(f"   - Tasks: {n_tasks}")
        print(f"   - Best Model: {metadata['best_model_name']}")
        print(f"   - Model Accuracy: {metadata['model_performance']['accuracy']:.4f}")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise e

# Request/Response Models
class PredictionRequest(BaseModel):
    completed_task_ids: List[int]

class PredictionResponse(BaseModel):
    predicted_task_id: int
    predicted_task_name: str
    confidence: Optional[float] = None
    top_3_predictions: Optional[List[Dict]] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_tasks: int
    model_accuracy: float

# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint for health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "n_tasks": n_tasks if n_tasks else 0,
        "model_accuracy": metadata['model_performance']['accuracy'] if metadata else 0
    }

@app.get("/health")
async def health_check():
    """Simple health check endpoint for cron jobs"""
    return {"status": "alive", "model_ready": model is not None}

@app.get("/tasks")
async def get_all_tasks():
    """Get all tasks with their details"""
    if task_df is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    tasks_list = []
    for idx, row in task_df.iterrows():
        tasks_list.append({
            "task_id": int(row['task_id']),
            "task_name": row['task_name'],
            "priority": int(row['priority']),
            "estimated_hours": float(row['estimated_hours']),
            "complexity": int(row['complexity']),
            "risk_level": int(row['risk_level']),
            "team_size": int(row['team_size']),
            "category": row['category'],
            "dependencies": dependencies.get(int(row['task_id']), [])
        })
    return {"tasks": tasks_list}

@app.get("/dependencies/{task_id}")
async def get_task_dependencies(task_id: int):
    """Get dependencies for a specific task"""
    if task_id not in dependencies:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": task_id,
        "task_name": task_names[task_id],
        "dependencies": dependencies[task_id],
        "dependency_names": [task_names[dep] for dep in dependencies[task_id]]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_next(request: PredictionRequest):
    """Predict the next task based on completed tasks"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    completed_ids = request.completed_task_ids
    
    # Validate input
    for task_id in completed_ids:
        if task_id < 0 or task_id >= n_tasks:
            raise HTTPException(status_code=400, detail=f"Invalid task ID: {task_id}")
    
    try:
        # Build features
        completed_set = set(completed_ids)
        
        # 1. Binary completion vector
        completion_vector = [1 if t in completed_set else 0 for t in range(n_tasks)]
        
        # 2. Aggregate features
        num_completed = len(completed_set)
        avg_priority = np.mean([task_df.loc[t, 'priority'] for t in completed_set]) if completed_set else 0
        avg_hours = np.mean([task_df.loc[t, 'estimated_hours'] for t in completed_set]) if completed_set else 0
        avg_complexity = np.mean([task_df.loc[t, 'complexity'] for t in completed_set]) if completed_set else 0
        total_hours = sum([task_df.loc[t, 'estimated_hours'] for t in completed_set])
        
        # 3. Category completion counts
        category_counts = [sum(1 for t in completed_set if task_df.loc[t, 'category'] == cat) for cat in category_names]
        
        # 4. Dependency metrics
        total_possible = len([t for t in range(n_tasks) if t not in completed_set])
        ready_tasks = sum(1 for t in range(n_tasks) if t not in completed_set 
                         and all(dep in completed_set for dep in dependencies[t]))
        dependency_ratio = ready_tasks / total_possible if total_possible > 0 else 1
        
        # 5. Remaining tasks metrics
        remaining_priorities = [task_df.loc[t, 'priority'] for t in range(n_tasks) if t not in completed_set]
        avg_remaining_priority = np.mean(remaining_priorities) if remaining_priorities else 0
        
        # 6. Blocked tasks
        blocked_tasks = sum(1 for t in range(n_tasks) if t not in completed_set 
                           and not all(dep in completed_set for dep in dependencies[t]))
        
        # Combine all features
        features = (completion_vector + 
                   [num_completed, avg_priority, avg_hours, avg_complexity, total_hours,
                    dependency_ratio, avg_remaining_priority, blocked_tasks] +
                   category_counts)
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Get prediction
        pred_id = model.predict(features_scaled)[0]
        pred_name = task_df.loc[pred_id, 'task_name']
        
        # Get confidence if available
        confidence = None
        top_3 = None
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_scaled)[0]
            confidence = float(proba[pred_id])
            
            # Get top 3 predictions
            top_3_indices = np.argsort(proba)[-3:][::-1]
            top_3 = [
                {
                    "task_id": int(idx),
                    "task_name": task_df.loc[idx, 'task_name'],
                    "confidence": float(proba[idx])
                }
                for idx in top_3_indices
            ]
        
        return PredictionResponse(
            predicted_task_id=int(pred_id),
            predicted_task_name=pred_name,
            confidence=confidence,
            top_3_predictions=top_3
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)