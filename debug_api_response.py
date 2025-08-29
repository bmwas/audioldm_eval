#!/usr/bin/env python3
"""
Quick debug script to see what the API is actually returning
"""

import requests
import json

API_BASE_URL = "http://10.4.11.192:2600"

def test_single_evaluation():
    """Test a single evaluation and print the raw response"""
    print("Testing API response structure...")
    
    # Create minimal test files
    import numpy as np
    import soundfile as sf
    from pathlib import Path
    import tempfile
    import os
    
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create test audio files
    sample_rate = 16000
    duration = 2  # seconds
    audio_data = np.random.randn(sample_rate * duration).astype(np.float32)
    
    gen_file = temp_dir / "test_gen.wav"
    ref_file = temp_dir / "test_ref.wav"
    
    sf.write(gen_file, audio_data, sample_rate)
    sf.write(ref_file, audio_data, sample_rate)
    
    try:
        # Make API request
        with open(gen_file, 'rb') as gf, open(ref_file, 'rb') as rf:
            files_data = [
                ('generated_files', ('test_gen.wav', gf, 'audio/wav')),
                ('reference_files', ('test_ref.wav', rf, 'audio/wav'))
            ]
            
            form_data = {
                'backbone': 'cnn14',
                'sampling_rate': 16000,
                'metrics': 'FAD,ISc'
            }
            
            print("Sending request to API...")
            response = requests.post(
                f"{API_BASE_URL}/evaluate",
                files=files_data,
                data=form_data,
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            
            print("\n" + "="*50)
            print("RAW API RESPONSE:")
            print("="*50)
            print(json.dumps(result, indent=2))
            print("="*50)
            
            # Check specific fields
            print(f"\nResponse Analysis:")
            print(f"- Status: {result.get('status')}")
            print(f"- Job ID: {result.get('job_id')}")
            print(f"- Has metrics: {'metrics' in result}")
            
            if 'metrics' in result:
                metrics = result['metrics']
                print(f"- Metrics is None: {metrics is None}")
                print(f"- Metrics type: {type(metrics)}")
                if metrics:
                    print(f"- Metrics keys: {list(metrics.keys())}")
                    print(f"- Metrics count: {len(metrics)}")
                    
                    print(f"\nMetrics values:")
                    for key, value in metrics.items():
                        print(f"  {key}: {value} (type: {type(value)})")
            
            return result
    
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    finally:
        # Cleanup
        try:
            os.remove(gen_file)
            os.remove(ref_file)
            os.rmdir(temp_dir)
        except:
            pass

if __name__ == "__main__":
    test_single_evaluation()
