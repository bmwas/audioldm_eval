#!/usr/bin/env python3
"""
AudioLDM Evaluation API

A FastAPI service that exposes all audioldm_eval metrics for audio generation evaluation.
Supports both paired and unpaired evaluation modes.
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import logging

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from audioldm_eval import EvaluationHelper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AudioLDM Evaluation API",
    description="API for evaluating audio generation models using audioldm_eval metrics",
    version="1.0.0"
)

# Global variables
UPLOAD_DIR = Path("/app/uploads")
RESULTS_DIR = Path("/app/results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Available backbone models
AVAILABLE_BACKBONES = ["cnn14", "mert"]

# Available metrics (from the README)
AVAILABLE_METRICS = [
    "FAD",  # Frechet Audio Distance
    "ISc",  # Inception Score  
    "FD",   # Frechet Distance
    "KID",  # Kernel Inception Distance
    "KL",   # KL Divergence
    "KL_Sigmoid",  # KL Divergence (sigmoid)
    "PSNR", # Peak Signal-to-Noise Ratio
    "SSIM", # Structural Similarity Index
    "LSD"   # Log-Spectral Distance
]

class EvaluationRequest(BaseModel):
    """Request model for evaluation"""
    backbone: str = "cnn14"
    sampling_rate: int = 16000
    limit_num: Optional[int] = None
    metrics: Optional[List[str]] = None  # If None, calculate all metrics

class EvaluationResponse(BaseModel):
    """Response model for evaluation results"""
    job_id: str
    status: str
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    evaluation_mode: Optional[str] = None  # "paired" or "unpaired"

# Store for tracking evaluation jobs
evaluation_jobs: Dict[str, EvaluationResponse] = {}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AudioLDM Evaluation API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/metrics": "Get available metrics",
            "/backbones": "Get available backbone models",
            "/evaluate": "Submit evaluation job",
            "/jobs/{job_id}": "Get evaluation results",
            "/docs": "API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        cuda_count = torch.cuda.device_count() if cuda_available else 0
        
        return {
            "status": "healthy",
            "cuda_available": cuda_available,
            "cuda_devices": cuda_count,
            "pytorch_version": torch.__version__
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/metrics")
async def get_available_metrics():
    """Get list of available evaluation metrics"""
    return {
        "available_metrics": AVAILABLE_METRICS,
        "recommended": ["FAD", "ISc"],
        "paired_only": ["KL", "KL_Sigmoid", "PSNR", "SSIM", "LSD"],
        "descriptions": {
            "FAD": "Frechet Audio Distance - Recommended for overall quality",
            "ISc": "Inception Score - Recommended for diversity evaluation", 
            "FD": "Frechet Distance using PANNs or MERT",
            "KID": "Kernel Inception Distance",
            "KL": "KL Divergence (softmax over logits) - Paired mode only",
            "KL_Sigmoid": "KL Divergence (sigmoid over logits) - Paired mode only",
            "PSNR": "Peak Signal-to-Noise Ratio - Paired mode only",
            "SSIM": "Structural Similarity Index - Paired mode only",
            "LSD": "Log-Spectral Distance - Paired mode only"
        }
    }

@app.get("/backbones")
async def get_available_backbones():
    """Get list of available backbone models"""
    return {
        "available_backbones": AVAILABLE_BACKBONES,
        "descriptions": {
            "cnn14": "PANNs CNN14 model - Recommended for general audio",
            "mert": "MERT model - Better for music understanding"
        }
    }

@app.post("/evaluate")
async def evaluate_audio(
    generated_files: List[UploadFile] = File(..., description="Generated audio files"),
    reference_files: List[UploadFile] = File(..., description="Reference audio files"),
    backbone: str = Form("cnn14", description="Backbone model to use"),
    sampling_rate: int = Form(16000, description="Audio sampling rate"),
    limit_num: Optional[int] = Form(None, description="Limit number of files to evaluate"),
    metrics: Optional[str] = Form(None, description="Comma-separated list of metrics (default: all)")
):
    """
    Evaluate audio generation quality
    
    Supports both paired and unpaired evaluation:
    - Paired: Same number of files with matching names
    - Unpaired: Different numbers or names
    
    Note: Some metrics (KL, KL_Sigmoid, PSNR, SSIM, LSD) only work in paired mode
    """
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    try:
        # Validate inputs
        if backbone not in AVAILABLE_BACKBONES:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid backbone '{backbone}'. Available: {AVAILABLE_BACKBONES}"
            )
        
        if not generated_files:
            raise HTTPException(status_code=400, detail="No generated files provided")
        
        if not reference_files:
            raise HTTPException(status_code=400, detail="No reference files provided")
        
        # Parse metrics list
        if metrics:
            requested_metrics = [m.strip() for m in metrics.split(",")]
            invalid_metrics = set(requested_metrics) - set(AVAILABLE_METRICS)
            if invalid_metrics:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid metrics: {invalid_metrics}. Available: {AVAILABLE_METRICS}"
                )
        else:
            requested_metrics = AVAILABLE_METRICS
        
        # Create temporary directories for this job
        job_dir = UPLOAD_DIR / job_id
        generated_dir = job_dir / "generated"
        reference_dir = job_dir / "reference"
        
        generated_dir.mkdir(parents=True, exist_ok=True)
        reference_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting evaluation job {job_id}")
        logger.info(f"Generated files: {len(generated_files)}, Reference files: {len(reference_files)}")
        
        # Save uploaded files
        generated_filenames = []
        for file in generated_files:
            if not file.filename:
                continue
            file_path = generated_dir / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            generated_filenames.append(file.filename)
        
        reference_filenames = []
        for file in reference_files:
            if not file.filename:
                continue
            file_path = reference_dir / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            reference_filenames.append(file.filename)
        
        # Determine evaluation mode (paired vs unpaired)
        evaluation_mode = "unpaired"
        if len(generated_filenames) == len(reference_filenames):
            # Check if filenames match (for paired evaluation)
            generated_set = set(generated_filenames)
            reference_set = set(reference_filenames)
            if generated_set == reference_set:
                evaluation_mode = "paired"
        
        logger.info(f"Evaluation mode: {evaluation_mode}")
        
        # Initialize evaluation job tracking
        evaluation_jobs[job_id] = EvaluationResponse(
            job_id=job_id,
            status="running",
            evaluation_mode=evaluation_mode
        )
        
        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize evaluator
        evaluator = EvaluationHelper(
            sampling_rate=sampling_rate,
            device=device,
            backbone=backbone
        )
        
        # Run evaluation
        logger.info("Running evaluation...")
        metrics_result = evaluator.main(
            generate_files_path=str(generated_dir),
            groundtruth_path=str(reference_dir),
            limit_num=limit_num
        )
        
        logger.info("Evaluation completed successfully")
        
        # Filter metrics if requested
        if metrics and requested_metrics != AVAILABLE_METRICS:
            filtered_result = {}
            for key, value in metrics_result.items():
                # Check if this metric was requested (case-insensitive)
                for requested in requested_metrics:
                    if requested.lower() in key.lower():
                        filtered_result[key] = value
                        break
            metrics_result = filtered_result
        
        # Update job status
        evaluation_jobs[job_id] = EvaluationResponse(
            job_id=job_id,
            status="completed",
            metrics=metrics_result,
            evaluation_mode=evaluation_mode
        )
        
        # Save results to disk
        result_file = RESULTS_DIR / f"{job_id}.json"
        with open(result_file, "w") as f:
            json.dump({
                "job_id": job_id,
                "metrics": metrics_result,
                "evaluation_mode": evaluation_mode,
                "backbone": backbone,
                "sampling_rate": sampling_rate,
                "limit_num": limit_num,
                "requested_metrics": requested_metrics,
                "generated_files_count": len(generated_filenames),
                "reference_files_count": len(reference_filenames)
            }, f, indent=2)
        
        # Cleanup temporary files
        shutil.rmtree(job_dir)
        
        return evaluation_jobs[job_id]
        
    except Exception as e:
        logger.error(f"Evaluation failed for job {job_id}: {e}")
        
        # Update job status with error
        evaluation_jobs[job_id] = EvaluationResponse(
            job_id=job_id,
            status="failed",
            error=str(e)
        )
        
        # Cleanup on error
        job_dir = UPLOAD_DIR / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir)
        
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/jobs/{job_id}")
async def get_evaluation_results(job_id: str):
    """Get evaluation results by job ID"""
    
    if job_id not in evaluation_jobs:
        # Check if results exist on disk
        result_file = RESULTS_DIR / f"{job_id}.json"
        if result_file.exists():
            with open(result_file, "r") as f:
                saved_result = json.load(f)
            return JSONResponse(content=saved_result)
        else:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return evaluation_jobs[job_id]

@app.get("/jobs")
async def list_jobs():
    """List all evaluation jobs"""
    # Get jobs from memory and disk
    all_jobs = dict(evaluation_jobs)
    
    # Add jobs from disk that aren't in memory
    for result_file in RESULTS_DIR.glob("*.json"):
        job_id = result_file.stem
        if job_id not in all_jobs:
            try:
                with open(result_file, "r") as f:
                    saved_result = json.load(f)
                all_jobs[job_id] = {
                    "job_id": job_id,
                    "status": "completed",
                    "evaluation_mode": saved_result.get("evaluation_mode"),
                    "backbone": saved_result.get("backbone"),
                    "metrics_count": len(saved_result.get("metrics", {}))
                }
            except Exception as e:
                logger.error(f"Error reading result file {result_file}: {e}")
    
    return {"jobs": list(all_jobs.values())}

if __name__ == "__main__":
    logger.info("Starting AudioLDM Evaluation API server...")
    logger.info(f"Available backbones: {AVAILABLE_BACKBONES}")
    logger.info(f"Available metrics: {AVAILABLE_METRICS}")
    
    # Run the server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=2600,
        log_level="info"
    )
