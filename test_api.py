#!/usr/bin/env python3
"""
AudioLDM Evaluation Test Script

This script demonstrates how to use the AudioLDM Evaluation library directly.
It creates sample audio files and tests the EvaluationHelper class.
"""

import os
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import json
from audioldm_eval import EvaluationHelper

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

def main():
    """Test AudioLDM Evaluation Helper directly"""
    
    print("AudioLDM Evaluation Test Script")
    print("=" * 50)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using CUDA device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    
    # Create test directories and files
    example_dir = Path("example")
    generation_result_path = example_dir / "paired"
    target_audio_path = example_dir / "reference"
    
    # Create sample audio files if they don't exist
    print("\n=== Setting up test data ===")
    if not generation_result_path.exists() or not target_audio_path.exists():
        print("Creating sample audio files for testing...")
        
        # Create directories
        generation_result_path.mkdir(parents=True, exist_ok=True)
        target_audio_path.mkdir(parents=True, exist_ok=True)
        
        # Create sample files
        num_files = 3
        duration = 2.0
        sr = 16000
        
        for i in range(num_files):
            # Create generated audio (sine wave with some noise)
            t = np.linspace(0, duration, int(sr * duration))
            freq = 440 + i * 100  # Different frequencies for each file
            generated_audio = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
            generated_audio = generated_audio.astype(np.float32)
            
            # Create reference audio (similar but slightly different)
            reference_audio = np.sin(2 * np.pi * (freq + 50) * t) + 0.05 * np.random.randn(len(t))
            reference_audio = reference_audio.astype(np.float32)
            
            # Save files with matching names for paired evaluation
            gen_file = generation_result_path / f"audio_{i:03d}.wav"
            ref_file = target_audio_path / f"audio_{i:03d}.wav"
            
            sf.write(gen_file, generated_audio, sr)
            sf.write(ref_file, reference_audio, sr)
            
            print(f"  Created: {gen_file.name} and {ref_file.name}")
    
    else:
        print("Using existing test files in example/ directory")
    
    # Test with different backbones
    backbones = ["cnn14", "mert"]
    
    for backbone in backbones:
        print(f"\n=== Testing with {backbone} backbone ===")
        
        try:
            # Initialize a helper instance
            # Note: EvaluationHelper now requires a backbone parameter
            evaluator = EvaluationHelper(
                sampling_rate=16000, 
                device=device,
                backbone=backbone  # `cnn14` refers to PANNs model, `mert` refers to MERT model
            )
            
            print(f"Evaluator initialized with backbone: {backbone}")
            print(f"Sampling rate: 16000")
            print(f"Device: {device}")
            
            # Perform evaluation, result will be print out and saved as json
            print("\nStarting evaluation...")
            metrics = evaluator.main(
                str(generation_result_path),
                str(target_audio_path),
                limit_num=None  # If you only intend to evaluate X (int) pairs of data, set limit_num=X
            )
            
            print(f"\n✅ Evaluation completed successfully with {backbone}!")
            print("\nMetrics Results:")
            print("-" * 30)
            
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    print(f"{metric_name:20}: {value:.6f}")
                else:
                    print(f"{metric_name:20}: {value}")
            
            # Save results to file
            results_file = f"evaluation_results_{backbone}.json"
            with open(results_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\nResults saved to: {results_file}")
            
        except Exception as e:
            print(f"❌ Evaluation failed with {backbone}: {str(e)}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Evaluation testing completed!")
    print(f"Test data location: {example_dir}")
    print(f"Generated files: {generation_result_path}")
    print(f"Reference files: {target_audio_path}")

if __name__ == "__main__":
    main()
