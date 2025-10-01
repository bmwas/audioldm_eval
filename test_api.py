#!/usr/bin/env python3
"""
AudioLDM Evaluation Test Script

This script demonstrates how to use the AudioLDM Evaluation API.
It creates sample audio files and tests the API endpoint.
"""

import os
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import json
import requests
import time
import argparse
import sys
from typing import Optional, List

def create_sample_audio_files(output_dir: Path, num_files: int = 1, duration: float = 2.0, sr: int = 16000):
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
        
        # Save files with specific names for API testing
        gen_file = output_dir / f"predicted_audio_{i+1}.wav"
        ref_file = output_dir / f"ref_audio_{i+1}.wav"
        
        sf.write(gen_file, generated_audio, sr)
        sf.write(ref_file, reference_audio, sr)
        
        generated_files.append(gen_file)
        reference_files.append(ref_file)
        
        print(f"  Created: {gen_file.name} and {ref_file.name}")
    
    return generated_files, reference_files

def call_evaluation_api(
    generated_files: List[Path], 
    reference_files: List[Path], 
    api_url: str = "http://localhost:2600",
    backbone: str = "cnn14",
    sampling_rate: int = 16000,
    metrics: Optional[List[str]] = None
) -> dict:
    """Call the evaluation API with the given files"""
    
    print(f"\n=== Calling Evaluation API ===")
    print(f"API URL: {api_url}")
    print(f"Backbone: {backbone}")
    print(f"Sampling Rate: {sampling_rate}")
    print(f"Generated files: {[f.name for f in generated_files]}")
    print(f"Reference files: {[f.name for f in reference_files]}")
    
    # Prepare the files for upload
    files = []
    
    # Add generated files
    for file_path in generated_files:
        if file_path.exists():
            files.append(('generated_files', (file_path.name, open(file_path, 'rb'), 'audio/wav')))
        else:
            raise FileNotFoundError(f"Generated file not found: {file_path}")
    
    # Add reference files
    for file_path in reference_files:
        if file_path.exists():
            files.append(('reference_files', (file_path.name, open(file_path, 'rb'), 'audio/wav')))
        else:
            raise FileNotFoundError(f"Reference file not found: {file_path}")
    
    # Prepare form data
    data = {
        'backbone': backbone,
        'sampling_rate': sampling_rate,
        'limit_num': None
    }
    
    # Add metrics if specified
    if metrics:
        data['metrics'] = ','.join(metrics)
    
    try:
        print(f"🔄 Sending request to API...")
        response = requests.post(
            f"{api_url}/evaluate",
            files=files,
            data=data,
            timeout=300  # 5 minute timeout
        )
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"✅ API call successful!")
                print(f"📊 Response type: {type(result)}")
                print(f"📊 Response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                return result
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON response: {e}")
                print(f"Raw response: {response.text[:500]}...")
                return {"error": f"JSON decode failed: {e}", "details": response.text}
        else:
            print(f"❌ API call failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return {"error": f"API call failed: {response.status_code}", "details": response.text}
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return {"error": f"Request failed: {str(e)}"}
    finally:
        # Close all file handles
        for _, (_, file_handle, _) in files:
            file_handle.close()

def check_api_health(api_url: str = "http://localhost:2600") -> bool:
    """Check if the API is running and healthy"""
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ API is healthy")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   CUDA available: {health_data.get('cuda_available', False)}")
            print(f"   Models preloaded: {health_data.get('models', {}).get('preloaded', False)}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def test_with_existing_files(
    predicted_file: str, 
    reference_file: str, 
    api_url: str = "http://localhost:2600",
    backbone: str = "cnn14"
) -> dict:
    """Test API with existing audio files"""
    
    print(f"Testing with existing files:")
    print(f"  Predicted: {predicted_file}")
    print(f"  Reference: {reference_file}")
    
    # Check if files exist
    pred_path = Path(predicted_file)
    ref_path = Path(reference_file)
    
    if not pred_path.exists():
        return {"error": f"Predicted file not found: {predicted_file}"}
    if not ref_path.exists():
        return {"error": f"Reference file not found: {reference_file}"}
    
    # Call the API
    return call_evaluation_api(
        generated_files=[pred_path],
        reference_files=[ref_path],
        api_url=api_url,
        backbone=backbone,
        sampling_rate=16000,
        metrics=None  # Request all metrics
    )

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AudioLDM Evaluation API Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with your own audio files
  python test_api.py --generated generated_audio.wav --reference reference_audio.wav
  
  # Test with multiple files
  python test_api.py --generated gen1.wav gen2.wav --reference ref1.wav ref2.wav
  
  # Test with specific backbone and metrics
  python test_api.py --generated gen.wav --reference ref.wav --backbone mert --metrics FAD,ISc
  
  # Test with custom API URL
  python test_api.py --generated gen.wav --reference ref.wav --api-url http://remote-server:2600
  
  # Create sample files and test (default behavior)
  python test_api.py
        """
    )
    
    parser.add_argument(
        "--generated", "-g",
        nargs="+",
        help="Generated audio file(s) to evaluate"
    )
    
    parser.add_argument(
        "--reference", "-r", 
        nargs="+",
        help="Reference audio file(s) for comparison"
    )
    
    parser.add_argument(
        "--api-url",
        default="http://localhost:2600",
        help="API server URL (default: http://localhost:2600)"
    )
    
    parser.add_argument(
        "--backbone",
        choices=["cnn14", "mert"],
        default="cnn14",
        help="Backbone model to use (default: cnn14)"
    )
    
    parser.add_argument(
        "--sampling-rate",
        type=int,
        choices=[16000, 32000],
        default=16000,
        help="Audio sampling rate (default: 16000)"
    )
    
    parser.add_argument(
        "--metrics",
        help="Comma-separated list of metrics to calculate (default: all metrics)"
    )
    
    parser.add_argument(
        "--output",
        help="Output file for results (default: auto-generated filename)"
    )
    
    parser.add_argument(
        "--create-samples",
        action="store_true",
        help="Create sample audio files for testing (ignores --generated and --reference)"
    )
    
    return parser.parse_args()

def validate_files(file_paths: List[str], file_type: str) -> List[Path]:
    """Validate that files exist and return Path objects"""
    validated_files = []
    
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            print(f"❌ {file_type} file not found: {file_path}")
            sys.exit(1)
        
        if not path.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a']:
            print(f"⚠️  Warning: {file_type} file may not be supported format: {file_path}")
        
        validated_files.append(path)
    
    return validated_files

def main():
    """Test AudioLDM Evaluation API"""
    
    args = parse_arguments()
    
    print("AudioLDM Evaluation API Test Script")
    print("=" * 50)
    
    # Check if API is running
    print(f"\n=== Checking API Health ===")
    if not check_api_health(args.api_url):
        print("❌ API is not running or not healthy. Please start the API server first.")
        print("   Run: python app.py")
        return
    
    # Determine which files to use
    if args.create_samples or (not args.generated and not args.reference):
        # Create sample files
        print("\n=== Creating sample audio files ===")
        example_dir = Path("example")
        example_dir.mkdir(exist_ok=True)
        
        generated_files, reference_files = create_sample_audio_files(
            example_dir, 
            num_files=1,
            duration=2.0,
            sr=args.sampling_rate
        )
        
        print(f"Generated files: {[f.name for f in generated_files]}")
        print(f"Reference files: {[f.name for f in reference_files]}")
        
    else:
        # Use provided files
        if not args.generated:
            print("❌ Error: --generated files are required when not creating samples")
            sys.exit(1)
        
        if not args.reference:
            print("❌ Error: --reference files are required when not creating samples")
            sys.exit(1)
        
        print(f"\n=== Using provided audio files ===")
        generated_files = validate_files(args.generated, "Generated")
        reference_files = validate_files(args.reference, "Reference")
        
        print(f"Generated files: {[f.name for f in generated_files]}")
        print(f"Reference files: {[f.name for f in reference_files]}")
    
    # Parse metrics if provided
    metrics_list = None
    if args.metrics:
        metrics_list = [m.strip() for m in args.metrics.split(",")]
        print(f"Requested metrics: {metrics_list}")
    
    # Call the API
    print(f"\n=== Testing with {args.backbone} backbone ===")
    
    try:
        result = call_evaluation_api(
            generated_files=generated_files,
            reference_files=reference_files,
            api_url=args.api_url,
            backbone=args.backbone,
            sampling_rate=args.sampling_rate,
            metrics=metrics_list
        )
        
        if result is None:
            print(f"❌ API call failed: No response received")
            return
        
        if "error" in result and result["error"] is not None:
            print(f"❌ API call failed: {result['error']}")
            if "details" in result:
                print(f"Details: {result['details']}")
            return
        
        # Display results
        print(f"\n✅ Evaluation completed successfully!")
        print("\nMetrics Results:")
        print("-" * 50)
        
        if "metrics" in result:
            metrics = result["metrics"]
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    if np.isinf(value) or np.isnan(value):
                        print(f"{metric_name:25}: {value} (⚠️  insufficient samples)")
                    else:
                        print(f"{metric_name:25}: {value:.6f}")
                else:
                    print(f"{metric_name:25}: {value}")
            
            # Check for warnings
            if "evaluation_warnings" in metrics:
                warnings = metrics["evaluation_warnings"]
                print(f"\n⚠️  Evaluation Warnings:")
                print(f"   {warnings.get('insufficient_samples', '')}")
                print(f"   Recommendation: {warnings.get('recommendation', '')}")
                if "affected_metrics" in warnings:
                    print(f"   Affected metrics: {warnings['affected_metrics']}")
        else:
            print("No metrics found in response")
        
        # Display job information
        print(f"\nJob Information:")
        print(f"   Job ID: {result.get('job_id', 'N/A')}")
        print(f"   Status: {result.get('status', 'N/A')}")
        print(f"   Evaluation Mode: {result.get('evaluation_mode', 'N/A')}")
        print(f"   Backbone: {args.backbone}")
        print(f"   Sampling Rate: {args.sampling_rate}")
        
        # Save results to file
        if args.output:
            results_file = args.output
        else:
            results_file = f"api_evaluation_results_{args.backbone}.json"
        
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return
    
    print("\n" + "=" * 50)
    print("API evaluation testing completed!")
    print(f"Generated files: {[f.name for f in generated_files]}")
    print(f"Reference files: {[f.name for f in reference_files]}")
    print(f"\nNote: Some metrics may show 'inf' values due to insufficient samples.")
    print(f"      This is expected behavior for single-file evaluation.")

if __name__ == "__main__":
    main()
