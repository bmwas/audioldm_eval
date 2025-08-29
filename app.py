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
import asyncio
from threading import Lock

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from audioldm_eval import EvaluationHelper

# Configure comprehensive logging with multiple loggers
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/audioldm_api.log', mode='a')
    ]
)

# Main logger
logger = logging.getLogger(__name__)

# Specialized loggers for different components
perf_logger = logging.getLogger(f"{__name__}.performance")
download_logger = logging.getLogger(f"{__name__}.downloads") 
model_logger = logging.getLogger(f"{__name__}.models")
api_logger = logging.getLogger(f"{__name__}.api")
system_logger = logging.getLogger(f"{__name__}.system")

# Helper function to get system stats
def get_system_stats():
    """Get current system resource usage"""
    try:
        import psutil
        process = psutil.Process()
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'memory_percent': process.memory_percent(),
            'disk_usage_gb': psutil.disk_usage('/').used / 1024 / 1024 / 1024
        }
    except ImportError:
        return {'error': 'psutil not available'}
    except Exception as e:
        return {'error': str(e)}

class ModelManager:
    """Manages preloaded EvaluationHelper instances for fast inference"""
    
    def __init__(self):
        self.models: Dict[str, EvaluationHelper] = {}
        self.lock = Lock()
        self.device = None
        self.preloaded = False
        
        # Initialize logging for this manager
        model_logger.info("🏗️ ModelManager initialized")
        system_stats = get_system_stats()
        system_logger.debug(f"Initial system stats: {system_stats}")
        
    async def preload_models(self):
        """Preload all backbone models on startup"""
        if self.preloaded:
            model_logger.warning("⚠️ Preloading already completed, skipping...")
            return
            
        preload_start = asyncio.get_event_loop().time()
        initial_stats = get_system_stats()
        model_logger.info("🚀 Starting model preloading...")
        system_logger.info(f"Pre-loading system stats: {initial_stats}")
        
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            model_logger.info(f"CUDA device detected: {device_props.name}")
            model_logger.debug(f"CUDA memory: {device_props.total_memory / 1024**3:.2f} GB")
        else:
            model_logger.warning("⚠️ CUDA not available, using CPU (this will be slow)")
        
        model_logger.info(f"Device selected: {self.device}")
        
        # Preload models for common configurations
        configs_to_preload = [
            ("cnn14", 16000),
            ("cnn14", 32000), 
            ("mert", 16000),
            ("mert", 32000),
        ]
        
        successful_loads = 0
        failed_loads = 0
        
        for i, (backbone, sampling_rate) in enumerate(configs_to_preload, 1):
            model_start_time = asyncio.get_event_loop().time()
            model_logger.info(f"📦 [{i}/{len(configs_to_preload)}] Loading {backbone} model (sr={sampling_rate})...")
            
            try:
                # Create model key
                model_key = f"{backbone}_{sampling_rate}"
                
                # Log memory before loading
                pre_load_stats = get_system_stats()
                perf_logger.debug(f"Pre-load stats for {model_key}: {pre_load_stats}")
                
                # Initialize EvaluationHelper (this downloads and loads the models)
                model_logger.info(f"🔄 Initializing EvaluationHelper for {backbone} (this may download models if not cached)...")
                evaluator = EvaluationHelper(
                    sampling_rate=sampling_rate,
                    device=self.device,
                    backbone=backbone
                )
                
                # Verify model is loaded and functional
                model_logger.debug(f"🔍 Verifying {backbone} model functionality...")
                # Models are loaded during __init__, so if we get here without exceptions, they're ready
                
                # Store the preloaded model
                with self.lock:
                    self.models[model_key] = evaluator
                    model_logger.debug(f"🔒 Model {model_key} stored in cache")
                
                # Log memory after loading
                post_load_stats = get_system_stats()
                perf_logger.debug(f"Post-load stats for {model_key}: {post_load_stats}")
                
                model_end_time = asyncio.get_event_loop().time()
                load_duration = model_end_time - model_start_time
                
                model_logger.info(f"✅ {backbone} model (sr={sampling_rate}) loaded and verified in {load_duration:.2f}s")
                perf_logger.info(f"Model {model_key} load time: {load_duration:.2f}s")
                
                successful_loads += 1
                
            except Exception as e:
                failed_loads += 1
                model_logger.error(f"❌ Failed to load {backbone} model (sr={sampling_rate}): {e}")
                model_logger.debug(f"Exception details for {backbone}:", exc_info=True)
                # Continue loading other models even if one fails
                continue
        
        preload_end = asyncio.get_event_loop().time()
        total_preload_time = preload_end - preload_start
        
        # Final system stats
        final_stats = get_system_stats()
        system_logger.info(f"Post-loading system stats: {final_stats}")
        
        self.preloaded = True
        
        # Comprehensive completion logging
        model_logger.info(f"🎉 Model preloading completed!")
        model_logger.info(f"📊 Summary: {successful_loads} successful, {failed_loads} failed")
        model_logger.info(f"⏱️ Total preload time: {total_preload_time:.2f}s")
        perf_logger.info(f"Preloading performance: {len(self.models)} models in {total_preload_time:.2f}s")
        
        if successful_loads > 0:
            model_logger.info(f"💾 Cached model configurations: {list(self.models.keys())}")
        else:
            model_logger.error("❌ No models were successfully loaded!")
        
    def get_evaluator(self, backbone: str, sampling_rate: int) -> EvaluationHelper:
        """Get a preloaded evaluator instance"""
        import time
        request_start = time.time()
        model_key = f"{backbone}_{sampling_rate}"
        
        model_logger.debug(f"🔍 Requesting evaluator: {model_key}")
        
        with self.lock:
            if model_key in self.models:
                request_end = time.time()
                retrieval_time = request_end - request_start
                model_logger.debug(f"✅ Retrieved preloaded model {model_key} in {retrieval_time:.4f}s")
                perf_logger.debug(f"Cache hit for {model_key}: {retrieval_time:.4f}s")
                return self.models[model_key]
        
        # Fallback: create new instance if not preloaded
        model_logger.warning(f"⚠️ Model {model_key} not preloaded, creating on-demand...")
        model_logger.info(f"🔄 Creating on-demand evaluator for {backbone} (sr={sampling_rate})...")
        
        try:
            evaluator = EvaluationHelper(
                sampling_rate=sampling_rate,
                device=self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                backbone=backbone
            )
            request_end = time.time()
            creation_time = request_end - request_start
            model_logger.warning(f"⚠️ On-demand model {model_key} created in {creation_time:.2f}s (preloading recommended)")
            perf_logger.warning(f"Cache miss for {model_key}: {creation_time:.2f}s")
            return evaluator
        except Exception as e:
            model_logger.error(f"❌ Failed to create on-demand model {model_key}: {e}")
            model_logger.debug(f"Exception details:", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to load model {model_key}: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about preloaded models"""
        model_logger.debug("📊 Generating model info report...")
        
        with self.lock:
            info = {
                "preloaded": self.preloaded,
                "device": str(self.device) if self.device else None,
                "loaded_models": list(self.models.keys()),
                "model_count": len(self.models),
                "memory_usage": get_system_stats()
            }
            
            model_logger.debug(f"Model info generated: {info}")
            return info

# Initialize model manager
model_manager = ModelManager()

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

@app.on_event("startup")
async def startup_event():
    """Preload models on application startup"""
    try:
        await model_manager.preload_models()
        logger.info("✅ Startup completed successfully")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        # Don't crash the server, continue with degraded functionality
        pass

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AudioLDM Evaluation API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check with model status",
            "/models": "Get preloaded model information", 
            "/metrics": "Get available metrics",
            "/backbones": "Get available backbone models",
            "/evaluate": "Submit evaluation job",
            "/jobs/{job_id}": "Get evaluation results",
            "/jobs": "List all evaluation jobs",
            "/docs": "API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with model status"""
    try:
        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        cuda_count = torch.cuda.device_count() if cuda_available else 0
        
        # Get model manager status
        model_info = model_manager.get_model_info()
        
        return {
            "status": "healthy",
            "cuda_available": cuda_available,
            "cuda_devices": cuda_count,
            "pytorch_version": torch.__version__,
            "models": model_info
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/models")
async def get_model_status():
    """Get detailed information about preloaded models"""
    try:
        model_info = model_manager.get_model_info()
        return {
            "model_manager": model_info,
            "available_backbones": AVAILABLE_BACKBONES,
            "supported_sampling_rates": [16000, 32000],
            "preloading_status": "completed" if model_info["preloaded"] else "pending"
        }
    except Exception as e:
        logger.error(f"Model status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model status check failed: {str(e)}")

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
    request_start = time.time()
    
    api_logger.info(f"🚀 NEW EVALUATION REQUEST - Job ID: {job_id}")
    api_logger.info(f"📝 Request params: backbone={backbone}, sr={sampling_rate}, limit={limit_num}")
    api_logger.debug(f"📊 Files received: {len(generated_files)} generated, {len(reference_files)} reference")
    
    # Log initial system stats
    initial_stats = get_system_stats()
    system_logger.debug(f"Pre-evaluation system stats for job {job_id}: {initial_stats}")
    
    try:
        # Validate inputs
        api_logger.debug(f"🔍 Validating request parameters for job {job_id}...")
        
        if backbone not in AVAILABLE_BACKBONES:
            api_logger.error(f"❌ Invalid backbone '{backbone}' for job {job_id}. Available: {AVAILABLE_BACKBONES}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid backbone '{backbone}'. Available: {AVAILABLE_BACKBONES}"
            )
        
        # Validate sampling rate
        if sampling_rate not in [16000, 32000]:
            api_logger.error(f"❌ Invalid sampling rate '{sampling_rate}' for job {job_id}. Supported: [16000, 32000]")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sampling rate '{sampling_rate}'. Supported: [16000, 32000]"
            )
        
        if not generated_files:
            api_logger.error(f"❌ No generated files provided for job {job_id}")
            raise HTTPException(status_code=400, detail="No generated files provided")
        
        if not reference_files:
            api_logger.error(f"❌ No reference files provided for job {job_id}")
            raise HTTPException(status_code=400, detail="No reference files provided")
        
        # Parse metrics list
        if metrics:
            requested_metrics = [m.strip() for m in metrics.split(",")]
            invalid_metrics = set(requested_metrics) - set(AVAILABLE_METRICS)
            if invalid_metrics:
                api_logger.error(f"❌ Invalid metrics for job {job_id}: {invalid_metrics}. Available: {AVAILABLE_METRICS}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid metrics: {invalid_metrics}. Available: {AVAILABLE_METRICS}"
                )
            api_logger.info(f"📊 Custom metrics requested for job {job_id}: {requested_metrics}")
        else:
            requested_metrics = AVAILABLE_METRICS
            api_logger.debug(f"📊 Using all available metrics for job {job_id}")
        
        api_logger.info(f"✅ Request validation passed for job {job_id}")
        
        # Create temporary directories for this job
        file_save_start = time.time()
        job_dir = UPLOAD_DIR / job_id
        generated_dir = job_dir / "generated"
        reference_dir = job_dir / "reference"
        
        api_logger.debug(f"📁 Creating temp directories for job {job_id}: {job_dir}")
        generated_dir.mkdir(parents=True, exist_ok=True)
        reference_dir.mkdir(parents=True, exist_ok=True)
        
        api_logger.info(f"🔄 Starting file upload processing for job {job_id}")
        api_logger.info(f"📊 File counts - Generated: {len(generated_files)}, Reference: {len(reference_files)}")
        
        # Save uploaded files
        generated_filenames = []
        total_generated_size = 0
        for i, file in enumerate(generated_files):
            if not file.filename:
                api_logger.warning(f"⚠️ Skipping generated file {i} with no filename in job {job_id}")
                continue
            file_path = generated_dir / file.filename
            api_logger.debug(f"💾 Saving generated file: {file.filename}")
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
                total_generated_size += len(content)
            generated_filenames.append(file.filename)
        
        reference_filenames = []
        total_reference_size = 0
        for i, file in enumerate(reference_files):
            if not file.filename:
                api_logger.warning(f"⚠️ Skipping reference file {i} with no filename in job {job_id}")
                continue
            file_path = reference_dir / file.filename
            api_logger.debug(f"💾 Saving reference file: {file.filename}")
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
                total_reference_size += len(content)
            reference_filenames.append(file.filename)
        
        file_save_end = time.time()
        file_save_duration = file_save_end - file_save_start
        
        api_logger.info(f"✅ File upload completed for job {job_id} in {file_save_duration:.2f}s")
        api_logger.info(f"📊 File sizes - Generated: {total_generated_size/1024/1024:.2f}MB, Reference: {total_reference_size/1024/1024:.2f}MB")
        perf_logger.info(f"File upload for job {job_id}: {file_save_duration:.2f}s, {(total_generated_size + total_reference_size)/1024/1024:.2f}MB total")
        
        # Determine evaluation mode (paired vs unpaired)
        api_logger.debug(f"🔍 Determining evaluation mode for job {job_id}...")
        evaluation_mode = "unpaired"
        if len(generated_filenames) == len(reference_filenames):
            # Check if filenames match (for paired evaluation)
            generated_set = set(generated_filenames)
            reference_set = set(reference_filenames)
            if generated_set == reference_set:
                evaluation_mode = "paired"
                api_logger.info(f"✅ Paired evaluation mode detected for job {job_id} (matching filenames)")
            else:
                api_logger.info(f"📊 Unpaired evaluation mode for job {job_id} (same count but different filenames)")
        else:
            api_logger.info(f"📊 Unpaired evaluation mode for job {job_id} (different file counts)")
        
        api_logger.info(f"🎯 Evaluation mode selected: {evaluation_mode}")
        
        # Initialize evaluation job tracking
        api_logger.debug(f"📝 Registering job {job_id} in tracking system...")
        evaluation_jobs[job_id] = EvaluationResponse(
            job_id=job_id,
            status="running",
            evaluation_mode=evaluation_mode
        )
        
        # Get preloaded evaluator (much faster than creating new instance)
        model_retrieval_start = time.time()
        api_logger.info(f"🔄 Retrieving evaluator for job {job_id}: {backbone} (sr={sampling_rate})")
        evaluator = model_manager.get_evaluator(backbone, sampling_rate)
        model_retrieval_end = time.time()
        model_retrieval_duration = model_retrieval_end - model_retrieval_start
        perf_logger.info(f"Model retrieval for job {job_id}: {model_retrieval_duration:.4f}s")
        
        # Run evaluation
        eval_start = time.time()
        api_logger.info(f"🚀 Starting evaluation computation for job {job_id}...")
        api_logger.debug(f"📊 Evaluation params - Generated dir: {generated_dir}, Reference dir: {reference_dir}, Limit: {limit_num}")
        
        # Log pre-evaluation system state
        pre_eval_stats = get_system_stats()
        system_logger.debug(f"Pre-evaluation system stats for job {job_id}: {pre_eval_stats}")
        
        metrics_result = evaluator.main(
            generate_files_path=str(generated_dir),
            groundtruth_path=str(reference_dir),
            limit_num=limit_num
        )
        
        eval_end = time.time()
        eval_duration = eval_end - eval_start
        
        # Log post-evaluation system state
        post_eval_stats = get_system_stats()
        system_logger.debug(f"Post-evaluation system stats for job {job_id}: {post_eval_stats}")
        
        api_logger.info(f"✅ Evaluation computation completed for job {job_id} in {eval_duration:.2f}s")
        perf_logger.info(f"Evaluation computation for job {job_id}: {eval_duration:.2f}s")
        api_logger.debug(f"📊 Metrics computed: {list(metrics_result.keys()) if metrics_result else 'None'}")
        
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
