#!/usr/bin/env python3
"""
AudioLDM Evaluation Web API Test Script

This script demonstrates how to use the AudioLDM Evaluation Web API.
It creates sample audio files and tests all available API endpoints.
"""

import os
import time
import requests
import numpy as np
import soundfile as sf
from pathlib import Path
import json
import argparse

# API Configuration
API_BASE_URL = "http://10.4.11.192:2600"

def create_sample_audio_files(output_dir: Path, num_files: int = 3, duration: float = 2.0, sr: int = 16000):
    """Create sample audio files for testing"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    reference_files = []
    
    print(f"Creating {num_files} sample audio files...")
    
    for i in range(num_files):
        # Create generated audio (sine wave with some noise)
        t = np.linspace(0, duration, int(sr * duration))
        freq = 440 + i * 100  # Different frequencies for each file
        generated_audio = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
        generated_audio = generated_audio.astype(np.float32)
        
        # Create reference audio (similar but slightly different)
        reference_audio = np.sin(2 * np.pi * (freq + 50) * t) + 0.05 * np.random.randn(len(t))
        reference_audio = reference_audio.astype(np.float32)
        
        # Save files
        gen_file = output_dir / f"generated_{i:03d}.wav"
        ref_file = output_dir / f"reference_{i:03d}.wav"
        
        sf.write(gen_file, generated_audio, sr)
        sf.write(ref_file, reference_audio, sr)
        
        generated_files.append(gen_file)
        reference_files.append(ref_file)
        
        print(f"  Created: {gen_file.name} and {ref_file.name}")
    
    return generated_files, reference_files

def test_health_endpoint():
    """Test the health check endpoint"""
    print("\n=== Testing Health Endpoint ===")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        health_data = response.json()
        
        print(f"Status: {health_data['status']}")
        print(f"CUDA Available: {health_data['cuda_available']}")
        print(f"CUDA Devices: {health_data['cuda_devices']}")
        print(f"PyTorch Version: {health_data['pytorch_version']}")
        
        return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_metrics_endpoint():
    """Test the metrics endpoint"""
    print("\n=== Testing Metrics Endpoint ===")
    try:
        response = requests.get(f"{API_BASE_URL}/metrics")
        response.raise_for_status()
        metrics_data = response.json()
        
        print(f"Available metrics: {metrics_data['available_metrics']}")
        print(f"Recommended: {metrics_data['recommended']}")
        print(f"Paired only: {metrics_data['paired_only']}")
        
        return metrics_data
    except Exception as e:
        print(f"Metrics endpoint failed: {e}")
        return None

def test_backbones_endpoint():
    """Test the backbones endpoint"""
    print("\n=== Testing Backbones Endpoint ===")
    try:
        response = requests.get(f"{API_BASE_URL}/backbones")
        response.raise_for_status()
        backbones_data = response.json()
        
        print(f"Available backbones: {backbones_data['available_backbones']}")
        for backbone, desc in backbones_data['descriptions'].items():
            print(f"  {backbone}: {desc}")
        
        return backbones_data
    except Exception as e:
        print(f"Backbones endpoint failed: {e}")
        return None

def test_evaluation_paired(generated_files, reference_files, backbone="cnn14", metrics=None):
    """Test paired evaluation"""
    print(f"\n=== Testing Paired Evaluation (backbone: {backbone}) ===")
    
    try:
        # Prepare files for upload (same names for paired evaluation)
        files_data = []
        
        # Generated files
        for i, file_path in enumerate(generated_files):
            # Rename for paired evaluation (same names)
            files_data.append(
                ('generated_files', (f'audio_{i:03d}.wav', open(file_path, 'rb'), 'audio/wav'))
            )
        
        # Reference files (same names as generated)
        for i, file_path in enumerate(reference_files):
            files_data.append(
                ('reference_files', (f'audio_{i:03d}.wav', open(file_path, 'rb'), 'audio/wav'))
            )
        
        # Prepare form data
        form_data = {
            'backbone': backbone,
            'sampling_rate': 16000,
            'limit_num': None
        }
        
        if metrics:
            form_data['metrics'] = ','.join(metrics)
        
        print("Uploading files and starting evaluation...")
        response = requests.post(
            f"{API_BASE_URL}/evaluate",
            files=files_data,
            data=form_data,
            timeout=300  # 5 minutes timeout
        )
        
        # Close file handles
        for _, file_tuple in files_data:
            if hasattr(file_tuple[1], 'close'):
                file_tuple[1].close()
        
        response.raise_for_status()
        result = response.json()
        
        print(f"Job ID: {result['job_id']}")
        print(f"Status: {result['status']}")
        print(f"Evaluation Mode: {result.get('evaluation_mode', 'unknown')}")
        
        if result['status'] == 'completed' and result.get('metrics'):
            print("\nMetrics Results:")
            for metric_name, value in result['metrics'].items():
                print(f"  {metric_name}: {value}")
        elif result['status'] == 'failed':
            print(f"Evaluation failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"Paired evaluation failed: {e}")
        return None

def test_evaluation_unpaired(generated_files, reference_files, backbone="cnn14"):
    """Test unpaired evaluation"""
    print(f"\n=== Testing Unpaired Evaluation (backbone: {backbone}) ===")
    
    try:
        # Prepare files for upload (different names for unpaired evaluation)
        files_data = []
        
        # Generated files
        for i, file_path in enumerate(generated_files):
            files_data.append(
                ('generated_files', (f'gen_{i:03d}.wav', open(file_path, 'rb'), 'audio/wav'))
            )
        
        # Reference files (different names, maybe different count)
        for i, file_path in enumerate(reference_files[:2]):  # Use only 2 reference files
            files_data.append(
                ('reference_files', (f'ref_{i:03d}.wav', open(file_path, 'rb'), 'audio/wav'))
            )
        
        # Prepare form data
        form_data = {
            'backbone': backbone,
            'sampling_rate': 16000,
            'metrics': 'FAD,ISc,FD'  # Only metrics that work in unpaired mode
        }
        
        print("Uploading files and starting unpaired evaluation...")
        response = requests.post(
            f"{API_BASE_URL}/evaluate",
            files=files_data,
            data=form_data,
            timeout=300
        )
        
        # Close file handles
        for _, file_tuple in files_data:
            if hasattr(file_tuple[1], 'close'):
                file_tuple[1].close()
        
        response.raise_for_status()
        result = response.json()
        
        print(f"Job ID: {result['job_id']}")
        print(f"Status: {result['status']}")
        print(f"Evaluation Mode: {result.get('evaluation_mode', 'unknown')}")
        
        if result['status'] == 'completed' and result.get('metrics'):
            print("\nMetrics Results:")
            for metric_name, value in result['metrics'].items():
                print(f"  {metric_name}: {value}")
        elif result['status'] == 'failed':
            print(f"Evaluation failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"Unpaired evaluation failed: {e}")
        return None

def test_job_status(job_id):
    """Test job status endpoint"""
    print(f"\n=== Testing Job Status for {job_id} ===")
    
    try:
        response = requests.get(f"{API_BASE_URL}/jobs/{job_id}")
        response.raise_for_status()
        job_data = response.json()
        
        print(f"Job ID: {job_data.get('job_id', 'N/A')}")
        print(f"Status: {job_data.get('status', 'N/A')}")
        
        if 'metrics' in job_data and job_data['metrics']:
            print(f"Metrics count: {len(job_data['metrics'])}")
        
        return job_data
        
    except Exception as e:
        print(f"Job status check failed: {e}")
        return None

def test_list_jobs():
    """Test list jobs endpoint"""
    print("\n=== Testing List Jobs ===")
    
    try:
        response = requests.get(f"{API_BASE_URL}/jobs")
        response.raise_for_status()
        jobs_data = response.json()
        
        print(f"Total jobs: {len(jobs_data['jobs'])}")
        for job in jobs_data['jobs'][:5]:  # Show first 5 jobs
            print(f"  Job {job.get('job_id', 'N/A')}: {job.get('status', 'N/A')}")
        
        return jobs_data
        
    except Exception as e:
        print(f"List jobs failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Test AudioLDM Evaluation Web API")
    parser.add_argument("--api-url", default="http://10.4.11.192:2600", help="API base URL")
    parser.add_argument("--num-files", type=int, default=3, help="Number of test files to create")
    parser.add_argument("--test-dir", default="./test_audio", help="Directory for test audio files")
    parser.add_argument("--skip-file-creation", action="store_true", help="Skip creating new test files")
    
    args = parser.parse_args()
    
    global API_BASE_URL
    API_BASE_URL = args.api_url
    
    print("AudioLDM Evaluation Web API Test Script")
    print("=" * 50)
    print(f"API URL: {API_BASE_URL}")
    
    # Test basic endpoints first
    if not test_health_endpoint():
        print("API is not healthy. Please check if the server is running.")
        return
    
    metrics_info = test_metrics_endpoint()
    backbones_info = test_backbones_endpoint()
    
    if not metrics_info or not backbones_info:
        print("Failed to get API configuration info.")
        return
    
    # Create or use existing test files
    test_dir = Path(args.test_dir)
    
    if not args.skip_file_creation:
        generated_files, reference_files = create_sample_audio_files(
            test_dir, 
            num_files=args.num_files
        )
    else:
        # Use existing files
        generated_files = list(test_dir.glob("generated_*.wav"))
        reference_files = list(test_dir.glob("reference_*.wav"))
        
        if not generated_files or not reference_files:
            print("No existing test files found. Creating new ones...")
            generated_files, reference_files = create_sample_audio_files(
                test_dir, 
                num_files=args.num_files
            )
    
    print(f"\nUsing {len(generated_files)} generated files and {len(reference_files)} reference files")
    
    # Test evaluations
    job_results = []
    
    # Test paired evaluation with different backbones
    for backbone in backbones_info['available_backbones']:
        result = test_evaluation_paired(
            generated_files, 
            reference_files, 
            backbone=backbone,
            metrics=['FAD', 'ISc']  # Quick test with just recommended metrics
        )
        if result:
            job_results.append(result)
    
    # Test unpaired evaluation
    result = test_evaluation_unpaired(generated_files, reference_files)
    if result:
        job_results.append(result)
    
    # Test job status for completed jobs
    for result in job_results:
        if result and 'job_id' in result:
            test_job_status(result['job_id'])
    
    # Test list jobs
    test_list_jobs()
    
    print("\n" + "=" * 50)
    print("Web API testing completed!")
    
    # Summary
    print(f"\nSummary:")
    print(f"- Created/used {len(generated_files)} generated and {len(reference_files)} reference audio files")
    print(f"- Submitted {len(job_results)} evaluation jobs")
    print(f"- Available backbones: {', '.join(backbones_info['available_backbones'])}")
    print(f"- Available metrics: {len(metrics_info['available_metrics'])}")
    
    print(f"\nTest files location: {test_dir}")
    print(f"API Documentation: {API_BASE_URL}/docs")

if __name__ == "__main__":
    main()
