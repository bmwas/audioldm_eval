# Audio Generation Evaluation

This toolbox aims to unify audio generation model evaluation for easier future comparison.

## Installation

### Method 1: Direct installation (if no issues)
```shell
pip install git+https://github.com/haoheliu/audioldm_eval
```

### Method 2: Virtual environment installation (Recommended)

If you encounter MySQL-python/ConfigParser issues during installation, use this method:

```shell
# Create and activate virtual environment (Python 3.10 recommended)
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install ssr_eval without dependencies to avoid MySQL-python issues
pip install ssr_eval --no-deps

# Clone and install the repository
git clone https://github.com/haoheliu/audioldm_eval.git
cd audioldm_eval
pip install -e . --no-deps

# Install remaining dependencies
pip install torch torchaudio transformers scikit-image torchlibrosa absl-py scipy tqdm librosa resampy
```

### Known Installation Issues

**Issue**: `ModuleNotFoundError: No module named 'ConfigParser'`
- **Cause**: The `ssr_eval` dependency chain includes `MySQL-python`, which is Python 2 only
- **Solution**: Use Method 2 above with `pip install ssr_eval --no-deps`

## Quick Start

After installation, generate test dataset:
```shell
python3 gen_test_file.py
```

Then perform a test run. A result for reference is attached [here](https://github.com/haoheliu/audioldm_eval/blob/main/example/paired_ref.json).
```shell
python3 test.py # Evaluate and save the json file to disk (example/paired.json)
```

## Evaluation metrics
We have the following metrics in this toolbox: 

- Recommanded:
  - FAD: Frechet audio distance
  - ISc: Inception score
- Other for references:
  - FD: Frechet distance, realized by either PANNs, a state-of-the-art audio classification model, or MERT, a music understanding model.
  - KID: Kernel inception score
  - KL: KL divergence (softmax over logits)
  - KL_Sigmoid: KL divergence (sigmoid over logits)
  - PSNR: Peak signal noise ratio
  - SSIM: Structural similarity index measure
  - LSD: Log-spectral distance

The evaluation function will accept the paths of two folders as main parameters. 
1. If two folder have **files with same name and same numbers of files**, the evaluation will run in **paired mode**.
2. If two folder have **different numbers of files or files with different name**, the evaluation will run in **unpaired mode**.

**These metrics will only be calculated in paried mode**: KL, KL_Sigmoid, PSNR, SSIM, LSD. 
In the unpaired mode, these metrics will return minus one.

## Evaluation on AudioCaps and AudioSet

The AudioCaps test set consists of audio files with multiple text annotations. To evaluate the performance of AudioLDM, we randomly selected one annotation per audio file, which can be found in the [accompanying json file](https://github.com/haoheliu/audioldm_eval/tree/c9e936ea538c4db7e971d9528a2d2eb4edac975d/example/AudioCaps).

Given the size of the AudioSet evaluation set with approximately 20,000 audio files, it may be impractical for audio generative models to perform evaluation on the entire set. As a result, we randomly selected 2,000 audio files for evaluation, with the corresponding annotations available in a [json file](https://github.com/haoheliu/audioldm_eval/tree/c9e936ea538c4db7e971d9528a2d2eb4edac975d/example/AudioSet).

For more information on our evaluation process, please refer to [our paper](https://arxiv.org/abs/2301.12503).

## Example

Single-GPU mode:

```python
import torch
from audioldm_eval import EvaluationHelper

# GPU acceleration is preferred
device = torch.device(f"cuda:{0}")

generation_result_path = "example/paired"
target_audio_path = "example/reference"

# Initialize a helper instance
# Note: EvaluationHelper now requires a backbone parameter
evaluator = EvaluationHelper(
    sampling_rate=16000, 
    device=device,
    backbone="cnn14"  # `cnn14` refers to PANNs model, `mert` refers to MERT model
)

# Perform evaluation, result will be print out and saved as json
metrics = evaluator.main(
    generation_result_path,
    target_audio_path,
    limit_num=None # If you only intend to evaluate X (int) pairs of data, set limit_num=X
)
```

Multi-GPU mode:

```python
import torch
from audioldm_eval import EvaluationHelperParallel
import torch.multiprocessing as mp

generation_result_path = "example/paired"
target_audio_path = "example/reference"

if __name__ == '__main__':    
    evaluator = EvaluationHelperParallel(
        sampling_rate=16000, 
        num_gpus=2,  # 2 denotes number of GPUs
        backbone="cnn14"  # `cnn14` refers to PANNs model, `mert` refers to MERT model
    )
    metrics = evaluator.main(
        generation_result_path,
        target_audio_path,
        limit_num=None # If you only intend to evaluate X (int) pairs of data, set limit_num=X
    )
```

You can use `CUDA_VISIBLE_DEVICES` to specify the GPU/GPUs to use.

```shell
CUDA_VISIBLE_DEVICES=0,1 python3 test.py
```

## Troubleshooting

### Common Installation Issues

**1. ModuleNotFoundError: No module named 'ConfigParser'**
```
ModuleNotFoundError: No module named 'ConfigParser'
```
- **Cause**: The `ssr_eval` dependency includes `MySQL-python`, which is Python 2 only and imports `ConfigParser` (renamed to `configparser` in Python 3)
- **Solution**: Use the virtual environment installation method above with `pip install ssr_eval --no-deps`

**2. TypeError: EvaluationHelper() missing required argument**
```
TypeError: EvaluationHelper.__init__() missing 1 required positional argument: 'backbone'
```
- **Cause**: The `EvaluationHelper` API has been updated to require a `backbone` parameter
- **Solution**: Update your code to include the backbone parameter:
  ```python
  # Old (no longer works)
  evaluator = EvaluationHelper(16000, device)
  
  # New (correct)
  evaluator = EvaluationHelper(sampling_rate=16000, device=device, backbone="cnn14")
  ```

### Supported Backbone Models
- `"cnn14"`: PANNs model (recommended)
- `"mert"`: MERT model for music understanding

## Docker API Usage

### Building the Docker Image

To build the Docker image for the AudioLDM Evaluation API:

```bash
# Build the Docker image
docker build -t ghcr.io/bmwas/audiollmtest:latest .

# Or build with custom base image
docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu22.04 -t ghcr.io/bmwas/audiollmtest:latest .
```

### Running the Docker Container

#### Basic Run (CPU only)
```bash
docker run -p 2600:2600 ghcr.io/bmwas/audiollmtest:latest
```

#### GPU-enabled Run (Recommended)
```bash
# For NVIDIA GPUs with Docker >= 19.03
docker run --gpus all -p 2600:2600 ghcr.io/bmwas/audiollmtest:latest

# Or specify a specific GPU
docker run --gpus device=0 -p 2600:2600 ghcr.io/bmwas/audiollmtest:latest

# With persistent volumes for uploads and results
docker run --gpus all -p 2600:2600 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/results:/app/results \
  ghcr.io/bmwas/audiollmtest:latest
```

#### Environment Variables
```bash
# Run with custom settings
docker run --gpus all -p 2600:2600 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTHONUNBUFFERED=1 \
  ghcr.io/bmwas/audiollmtest:latest
```

### API Endpoints

Once the container is running, the API will be available at `http://localhost:2600`

#### Available Endpoints:

- **GET** `/` - API information and available endpoints
- **GET** `/health` - Health check with CUDA status
- **GET** `/metrics` - List all available evaluation metrics
- **GET** `/backbones` - List available backbone models
- **POST** `/evaluate` - Submit audio files for evaluation
- **GET** `/jobs/{job_id}` - Get evaluation results by job ID
- **GET** `/jobs` - List all evaluation jobs
- **GET** `/docs` - Interactive API documentation (Swagger UI)

#### Example API Usage

**1. Check API Health:**
```bash
curl http://localhost:2600/health
```

**2. Get Available Metrics:**
```bash
curl http://localhost:2600/metrics
```

**3. Submit Evaluation Job:**
```bash
# Upload audio files for evaluation
curl -X POST "http://localhost:2600/evaluate" \
  -F "generated_files=@path/to/generated1.wav" \
  -F "generated_files=@path/to/generated2.wav" \
  -F "reference_files=@path/to/reference1.wav" \
  -F "reference_files=@path/to/reference2.wav" \
  -F "backbone=cnn14" \
  -F "sampling_rate=16000" \
  -F "metrics=FAD,ISc,FD"
```

**4. Check Job Status:**
```bash
# Replace JOB_ID with the actual job ID returned from evaluation
curl http://localhost:2600/jobs/JOB_ID
```

### Testing the API

Use the provided test script to verify the API is working correctly:

```bash
# Install test dependencies (if not in virtual environment)
pip install requests soundfile numpy

# Run the test script
python test_api.py

# Or use the test runner script
./run_test.sh

# Test with custom parameters
python test_api.py --api-url http://localhost:2600 --num-files 5
```

### API Features

- **Paired and Unpaired Evaluation**: Automatically detects evaluation mode based on file names and counts
- **All AudioLDM Metrics**: Supports FAD, ISc, FD, KID, KL, KL_Sigmoid, PSNR, SSIM, LSD
- **Multiple Backbones**: Choose between CNN14 (PANNs) and MERT models
- **Batch Processing**: Upload multiple files in a single request
- **Job Tracking**: Persistent job storage with unique IDs
- **CUDA Support**: GPU acceleration when available
- **Interactive Documentation**: Swagger UI at `/docs` endpoint

### Metric Limitations

- **Paired Mode Only**: KL, KL_Sigmoid, PSNR, SSIM, LSD metrics require paired evaluation (same number of files with matching names)
- **Unpaired Mode**: FAD, ISc, FD, KID work with different numbers of files or mismatched names
- **Recommended Metrics**: FAD (quality) and ISc (diversity) are recommended for most use cases

## Note
- **Installation and API Updates:**
  - **Installation Issues**: If you encounter `ModuleNotFoundError: No module named 'ConfigParser'`, use the virtual environment installation method with `pip install ssr_eval --no-deps`
  - **EvaluationHelper API Change**: The `EvaluationHelper` class now requires a `backbone` parameter in its constructor. Use `backbone="cnn14"` for PANNs model or `backbone="mert"` for MERT model.
- Update on 29 Sept 2024:
  - **MERT inference:** Note that the MERT model is trained on 24 kHz, but the repository inference in either 16 kHz or 32 kHz mode. In both modes, we resample the audio to 24 kHz.
  - **FAD calculation:** The FAD calculation currently even in the parallel mode will only be done on the first GPU, due to the implementation we currently use.
- Update on 24 June 2023: 
  - **Issues on model evaluation:** I found the PANNs based Frechet Distance and KL score is not as robust as FAD sometimes. For example, when the generation are all silent audio, the FAD and KL still indicate model perform very well, while FAD and Inception Score (IS) can still reflect the model true bad performance. Sometimes the resample method on audio can significantly affect the FD (+-30) and KL (+-0.4) performance as well.
    - To address this issue, in another branch of this repo ([passt_replace_panns](https://github.com/haoheliu/audioldm_eval/tree/passt_replace_panns)), I change the PANNs model to Passt, which I found to be more robust to resample method and other trival mismatches.

  - **Update on code:** The calculation of FAD is slow. Now, after each calculation of a folder, the code will save the FAD feature into an .npy file for later reference. 

## 📊 API Metric Interpretation

### Expected NaN Values
Some metrics may return `nan` values in certain scenarios - this is **normal behavior**:

- **PSNR, SSIM, LSD**: Only available in **paired mode** (same filenames)
- **KL, KL_Sigmoid**: Only available in **paired mode** 
- **ISc (Inception Score)**: May return `nan` with very small sample sizes (< 3 files)
- **KID**: May return `nan` with insufficient data

### Successful Evaluation Example
```json
{
  "job_id": "abc123...",
  "status": "completed",
  "metrics": {
    "frechet_audio_distance": 4.62015,
    "frechet_distance": 15.50206,
    "kullback_leibler_divergence_sigmoid": 0.00214,
    "kullback_leibler_divergence_softmax": 0.25198,
    "inception_score_mean": 1.01896,
    "inception_score_std": 0.005161
  },
  "evaluation_mode": "paired"
}
```

## ⚠️ Expected Warnings

You may see these **harmless** deprecation warnings in the logs:
```
UserWarning: In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec
UserWarning: torio.io._streaming_media_decoder.StreamingMediaDecoder has been deprecated
```

These warnings are from TorchAudio and **do not affect functionality**. They can be safely ignored.

## 🚀 Performance Expectations

### Model Preloading (First Startup)
- **Initial startup**: 60-120 seconds (downloads and loads all models)
- **Subsequent startups**: 40-80 seconds (loads cached models)

### API Response Times
- **Preloaded models**: < 1 second model retrieval
- **Evaluation computation**: 15-45 seconds (depends on audio length and file count)
- **File upload**: Varies by file size and network

### Expected Startup Logs
```
INFO:app.models - 🚀 Starting model preloading...
INFO:app.models - CUDA device detected: NVIDIA RTX 4090
INFO:app.models - 📦 [1/4] Loading cnn14 model (sr=16000)...
INFO:app.models - ✅ cnn14 model (sr=16000) loaded and verified in 20.11s
INFO:app.models - 🎉 Model preloading completed!
INFO:app.models - 📊 Summary: 4 successful, 0 failed
```

## 📝 Comprehensive Logging

The API includes extensive logging for monitoring and debugging:

### Log Locations
- **Console output**: Real-time monitoring
- **Log file**: `/tmp/audioldm_api.log` (persistent storage)

### Log Categories
- **🚀 API Requests**: Request lifecycle and performance
- **📦 Model Operations**: Loading, caching, and retrieval
- **⬇️ Downloads**: Model download progress and verification
- **📊 Performance**: Timing measurements and resource usage
- **🔍 System**: Resource monitoring and health checks

### Monitoring Examples
```bash
# Monitor API logs in real-time
tail -f /tmp/audioldm_api.log

# Check model preloading status
curl http://10.4.11.192:2600/models

# View system health
curl http://10.4.11.192:2600/health
```

## 🔧 Production Deployment Notes

### Resource Requirements
- **GPU Memory**: 4GB+ recommended for model preloading
- **RAM**: 8GB+ recommended for concurrent requests
- **Disk Space**: 2GB+ for cached models

### Performance Optimization
1. **Model Preloading**: Always enabled by default for fast responses
2. **Caching**: Models cached to `~/.cache/audioldm_eval/ckpt/`
3. **Concurrent Requests**: API handles multiple evaluation jobs
4. **Resource Monitoring**: Built-in system stats tracking

### Health Monitoring
```bash
# Check API health
curl http://10.4.11.192:2600/health

# Expected healthy response:
{
  "status": "healthy",
  "cuda_available": true,
  "cuda_devices": 1,
  "pytorch_version": "2.8.0+cu128",
  "models": {
    "preloaded": true,
    "model_count": 4,
    "loaded_models": ["cnn14_16000", "cnn14_32000", "mert_16000", "mert_32000"]
  }
}
```

## TODO

- [ ] Add pretrained AudioLDM model.
- [ ] Add CLAP score

## Cite this repo

If you found this tool useful, please consider citing
```bibtex
@article{audioldm2-2024taslp,
  author={Liu, Haohe and Yuan, Yi and Liu, Xubo and Mei, Xinhao and Kong, Qiuqiang and Tian, Qiao and Wang, Yuping and Wang, Wenwu and Wang, Yuxuan and Plumbley, Mark D.},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={AudioLDM 2: Learning Holistic Audio Generation With Self-Supervised Pretraining}, 
  year={2024},
  volume={32},
  pages={2871-2883},
  doi={10.1109/TASLP.2024.3399607}
}

@article{liu2023audioldm,
  title={{AudioLDM}: Text-to-Audio Generation with Latent Diffusion Models},
  author={Liu, Haohe and Chen, Zehua and Yuan, Yi and Mei, Xinhao and Liu, Xubo and Mandic, Danilo and Wang, Wenwu and Plumbley, Mark D},
  journal={Proceedings of the International Conference on Machine Learning},
  year={2023}
  pages={21450-21474}
}
```

## Reference

> https://github.com/toshas/torch-fidelity

> https://github.com/v-iashin/SpecVQGAN 
