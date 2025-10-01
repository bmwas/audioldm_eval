# test_api.py Usage Examples

The updated `test_api.py` script now accepts both reference and generated audio files as command-line inputs.

## Basic Usage

### 1. Test with your own audio files
```bash
python test_api.py --generated predicted_audio_1.wav --reference ref_audio_1.wav
```

### 2. Test with multiple files
```bash
python test_api.py --generated gen1.wav gen2.wav --reference ref1.wav ref2.wav
```

### 3. Test with specific backbone and metrics
```bash
python test_api.py --generated gen.wav --reference ref.wav --backbone mert --metrics FAD,ISc
```

### 4. Test with custom API URL
```bash
python test_api.py --generated gen.wav --reference ref.wav --api-url http://remote-server:2600
```

### 5. Create sample files and test (default behavior)
```bash
python test_api.py
```

## Command Line Options

- `--generated, -g`: Generated audio file(s) to evaluate
- `--reference, -r`: Reference audio file(s) for comparison  
- `--api-url`: API server URL (default: http://localhost:2600)
- `--backbone`: Backbone model to use - cnn14 or mert (default: cnn14)
- `--sampling-rate`: Audio sampling rate - 16000 or 32000 (default: 16000)
- `--metrics`: Comma-separated list of metrics to calculate (default: all metrics)
- `--output`: Output file for results (default: auto-generated filename)
- `--create-samples`: Create sample audio files for testing

## Available Metrics

- FAD: Frechet Audio Distance
- ISc: Inception Score
- FD: Frechet Distance
- KID: Kernel Inception Distance
- KL: KL Divergence (softmax)
- KL_Sigmoid: KL Divergence (sigmoid)
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index
- LSD: Log-Spectral Distance

## Example Output

```
AudioLDM Evaluation API Test Script
==================================================

=== Checking API Health ===
✅ API is healthy
   Status: healthy
   CUDA available: True
   Models preloaded: True

=== Using provided audio files ===
Generated files: ['predicted_audio_1.wav']
Reference files: ['ref_audio_1.wav']

=== Testing with cnn14 backbone ===
🔄 Sending request to API...
📊 Response status: 200
✅ API call successful!

✅ Evaluation completed successfully!

Metrics Results:
--------------------------------------------------
frechet_audio_distance    : 4.620150
frechet_distance          : 15.502060
kullback_leibler_divergence_sigmoid: 0.002140
kullback_leibler_divergence_softmax: 0.251980
inception_score_mean      : 1.018960
inception_score_std       : 0.005161
psnr                      : 28.456000
ssim                      : 0.789000
lsd                       : 2.134000

Job Information:
   Job ID: abc123-def456-ghi789
   Status: completed
   Evaluation Mode: paired
   Backbone: cnn14
   Sampling Rate: 16000

Results saved to: api_evaluation_results_cnn14.json
```
