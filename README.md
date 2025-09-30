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

## 📚 Model Sources & Citations

### **CNN14 Models (`cnn14_16000`, `cnn14_32000`)**

#### **Source & Download Locations:**
- **16kHz Model**: `Cnn14_16k_mAP=0.438.pth`
  - **Download URL**: `https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth`
  - **Zenodo Record**: [3987831](https://zenodo.org/record/3987831)
- **32kHz Model**: `Cnn14_mAP=0.431.pth`
  - **Download URL**: `https://zenodo.org/record/3576403/files/Cnn14_mAP%3D0.431.pth`
  - **Zenodo Record**: [3576403](https://zenodo.org/record/3576403)

#### **Paper Citation:**
```bibtex
@article{kong2020panns,
  title={PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition},
  author={Kong, Qiuqiang and Cao, Yuxuan and Iqbal, Turab and Wang, Yuxuan and Wang, Wenwu and Plumbley, Mark D},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={28},
  pages={2880--2894},
  year={2020},
  publisher={IEEE}
}
```

#### **Model Details:**
- **Architecture**: CNN14 (14-layer Convolutional Neural Network)
- **Training Data**: AudioSet (large-scale audio dataset)
- **Classes**: 527 audio event classes
- **Performance**: mAP=0.431 (32kHz), mAP=0.438 (16kHz)
- **Framework**: PANNs (Pretrained Audio Neural Networks)

### **MERT Models (`mert_16000`, `mert_32000`)**

#### **Source & Download Location:**
- **Hugging Face Model**: `m-a-p/MERT-v1-95M`
- **Model Hub**: [https://huggingface.co/m-a-p/MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M)
- **Download Method**: Automatic via `transformers` library

#### **Paper Citation:**
```bibtex
@article{li2023mert,
  title={MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training},
  author={Li, Yizhi and Yuan, Ruibin and Zhang, Ge and Ma, Yinghao and Chen, Xingran and Yin, Hanzhi and Huang, Haotian and Ni, Chenghao and Weng, Wenhao and Liu, Xubo and others},
  journal={arXiv preprint arXiv:2306.00107},
  year={2023}
}
```

#### **Model Details:**
- **Architecture**: Transformer-based encoder (95M parameters)
- **Training Data**: Large-scale music datasets
- **Target Sample Rate**: 24kHz (automatically resampled from 16kHz/32kHz)
- **Specialization**: Music understanding and representation learning
- **Framework**: MERT (Music Encoder Representations from Transformers)

### **Model Loading Process:**
```python
# CNN14 Models (from codebase)
if sample_rate == 16000:
    state_dict = torch.load("~/.cache/audioldm_eval/ckpt/Cnn14_16k_mAP=0.438.pth")
elif sample_rate == 32000:
    state_dict = torch.load("~/.cache/audioldm_eval/ckpt/Cnn14_mAP=0.431.pth")

# MERT Model (from codebase)
self.mel_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
```

### **Model Configurations:**
- **CNN14 16kHz**: window_size=512, hop_size=160, mel_bins=64, fmin=50, fmax=8000
- **CNN14 32kHz**: window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000
- **MERT**: Automatically handles resampling to 24kHz regardless of input rate

### **Usage Recommendations:**

#### **When to Use CNN14:**
- **General audio evaluation** (speech, environmental sounds, etc.)
- **AudioSet-based tasks** (527 audio event classes)
- **Standard audio classification** benchmarks

#### **When to Use MERT:**
- **Music-specific evaluation** (better music understanding)
- **Musical content analysis** (harmony, melody, rhythm)
- **Music generation quality assessment**

The models are automatically downloaded and cached in `~/.cache/audioldm_eval/ckpt/` on first use, ensuring reproducible evaluation results across different systems.

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

## 📡 Comprehensive API Usage Guide

### 🌐 Base URL Configuration
```bash
# Local development
BASE_URL="http://localhost:2600"

# Remote server (replace with your server IP)
BASE_URL="http://your-server-ip:2600"
```

### 📋 Detailed API Calls

#### **1. Health Check**
```bash
curl http://localhost:2600/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "cuda_available": true,
  "cuda_devices": 1,
  "pytorch_version": "2.8.0+cu128",
  "models": {
    "preloaded": true,
    "device": "cuda:0",
    "loaded_models": ["cnn14_16000", "cnn14_32000", "mert_16000", "mert_32000"],
    "model_count": 4,
    "memory_usage": {
      "cpu_percent": 15.2,
      "memory_mb": 2048.5,
      "memory_percent": 25.1
    }
  }
}
```

#### **2. Get Available Metrics**
```bash
curl http://localhost:2600/metrics
```

**Expected Response:**
```json
{
  "available_metrics": ["FAD", "ISc", "FD", "KID", "KL", "KL_Sigmoid", "PSNR", "SSIM", "LSD"],
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
```

#### **3. Get Available Backbones**
```bash
curl http://localhost:2600/backbones
```

**Expected Response:**
```json
{
  "available_backbones": ["cnn14", "mert"],
  "descriptions": {
    "cnn14": "PANNs CNN14 model - Recommended for general audio",
    "mert": "MERT model - Better for music understanding"
  }
}
```

#### **4. Get Model Status**
```bash
curl http://localhost:2600/models
```

**Expected Response:**
```json
{
  "model_manager": {
    "preloaded": true,
    "device": "cuda:0",
    "loaded_models": ["cnn14_16000", "cnn14_32000", "mert_16000", "mert_32000"],
    "model_count": 4,
    "memory_usage": {
      "cpu_percent": 15.2,
      "memory_mb": 2048.5
    }
  },
  "available_backbones": ["cnn14", "mert"],
  "supported_sampling_rates": [16000, 32000],
  "preloading_status": "completed"
}
```

### 🎯 Main Evaluation Endpoint

#### **Basic Evaluation (All Defaults)**
```bash
curl -X POST "http://localhost:2600/evaluate" \
  -F "generated_files=@generated1.wav" \
  -F "generated_files=@generated2.wav" \
  -F "reference_files=@reference1.wav" \
  -F "reference_files=@reference2.wav"
```

#### **Advanced Evaluation with All Parameters**
```bash
curl -X POST "http://localhost:2600/evaluate" \
  -F "generated_files=@generated1.wav" \
  -F "generated_files=@generated2.wav" \
  -F "generated_files=@generated3.wav" \
  -F "reference_files=@reference1.wav" \
  -F "reference_files=@reference2.wav" \
  -F "reference_files=@reference3.wav" \
  -F "backbone=cnn14" \
  -F "sampling_rate=16000" \
  -F "metrics=FAD,ISc,FD,KID" \
  -F "limit_num=10"
```

### 📊 Evaluation Parameters Explained

#### **Required Parameters:**
- **`generated_files`** (List[UploadFile]): Generated audio files to evaluate
  - Supports multiple files in single request
  - Common formats: WAV, MP3, FLAC, M4A
  - No size limit (but larger files take longer to process)

- **`reference_files`** (List[UploadFile]): Reference/ground truth audio files
  - Should match the content type of generated files
  - Used for comparison and metric calculation

#### **Optional Parameters:**
- **`backbone`** (str, default="cnn14"): Model backbone to use
  - `"cnn14"`: PANNs CNN14 model (recommended for general audio)
  - `"mert"`: MERT model (better for music understanding)

- **`sampling_rate`** (int, default=16000): Audio sampling rate
  - `16000`: Standard for speech audio
  - `32000`: Higher quality, better for music
  - Audio is automatically resampled to match model requirements

- **`metrics`** (str, optional): Comma-separated list of metrics to calculate
  - If not specified, all available metrics are calculated
  - Example: `"FAD,ISc,FD"` or `"FAD,ISc,KL,PSNR"`
  - See metric descriptions below for details

- **`limit_num`** (int, optional): Limit number of file pairs to evaluate
  - Useful for testing with large datasets
  - If not specified, all uploaded files are processed

### 🎵 Evaluation Modes

The API automatically detects evaluation mode based on uploaded files:

#### **Paired Mode** (Same filenames and count)
```bash
# Files with matching names
generated1.wav ↔ reference1.wav
generated2.wav ↔ reference2.wav
generated3.wav ↔ reference3.wav
```
- **Available Metrics**: All metrics (FAD, ISc, FD, KID, KL, KL_Sigmoid, PSNR, SSIM, LSD)
- **Use Case**: Direct comparison between generated and reference audio

#### **Unpaired Mode** (Different filenames or counts)
```bash
# Different filenames or counts
generated_audio_1.wav, generated_audio_2.wav
reference_sound_1.wav, reference_sound_2.wav, reference_sound_3.wav
```
- **Available Metrics**: FAD, ISc, FD, KID only
- **Use Case**: Comparing overall quality/diversity of audio sets

### 📈 Metric Descriptions & Interpretation

#### **Recommended Metrics (Use These First)**

**FAD (Frechet Audio Distance)**
- **Range**: 0 to ∞ (lower is better)
- **Interpretation**: 
  - < 1.0: Excellent quality
  - 1.0-3.0: Good quality
  - 3.0-10.0: Moderate quality
  - > 10.0: Poor quality
- **Use Case**: Overall audio quality assessment

**ISc (Inception Score)**
- **Range**: 1.0 to ∞ (higher is better)
- **Interpretation**:
  - 1.0-2.0: Low diversity
  - 2.0-5.0: Moderate diversity
  - 5.0-10.0: Good diversity
  - > 10.0: High diversity
- **Use Case**: Audio diversity and variety assessment

#### **Additional Quality Metrics**

**FD (Frechet Distance)**
- **Range**: 0 to ∞ (lower is better)
- **Interpretation**: Similar to FAD but using different feature extractor
- **Use Case**: Alternative quality metric for comparison

**KID (Kernel Inception Distance)**
- **Range**: 0 to ∞ (lower is better)
- **Interpretation**: Unbiased alternative to FAD
- **Use Case**: More stable metric for small datasets

#### **Paired-Only Metrics** (Require matching filenames)

**KL (Kullback-Leibler Divergence)**
- **Range**: 0 to ∞ (lower is better)
- **Interpretation**: 
  - < 0.1: Very similar distributions
  - 0.1-1.0: Similar distributions
  - > 1.0: Different distributions
- **Use Case**: Statistical similarity between audio features

**PSNR (Peak Signal-to-Noise Ratio)**
- **Range**: 0 to ∞ dB (higher is better)
- **Interpretation**:
  - > 40 dB: Excellent quality
  - 30-40 dB: Good quality
  - 20-30 dB: Acceptable quality
  - < 20 dB: Poor quality
- **Use Case**: Signal fidelity measurement

**SSIM (Structural Similarity Index)**
- **Range**: -1 to 1 (higher is better)
- **Interpretation**:
  - > 0.9: Excellent similarity
  - 0.7-0.9: Good similarity
  - 0.5-0.7: Moderate similarity
  - < 0.5: Poor similarity
- **Use Case**: Perceptual similarity measurement

**LSD (Log-Spectral Distance)**
- **Range**: 0 to ∞ (lower is better)
- **Interpretation**:
  - < 1.0: Very similar spectral content
  - 1.0-3.0: Similar spectral content
  - > 3.0: Different spectral content
- **Use Case**: Spectral similarity measurement

### 📤 Expected Response Format

#### **Successful Evaluation Response:**
```json
{
  "job_id": "abc123-def456-ghi789",
  "status": "completed",
  "metrics": {
    "frechet_audio_distance": 4.62015,
    "frechet_distance": 15.50206,
    "kullback_leibler_divergence_sigmoid": 0.00214,
    "kullback_leibler_divergence_softmax": 0.25198,
    "inception_score_mean": 1.01896,
    "inception_score_std": 0.005161,
    "kernel_inception_distance": 0.12345,
    "peak_signal_to_noise_ratio": 28.456,
    "structural_similarity_index": 0.789,
    "log_spectral_distance": 2.134
  },
  "evaluation_mode": "paired"
}
```

#### **Error Response:**
```json
{
  "job_id": "abc123-def456-ghi789",
  "status": "failed",
  "error": "Invalid backbone 'invalid_model'. Available: ['cnn14', 'mert']"
}
```

### 🔍 Job Status Checking

#### **Get Specific Job Results:**
```bash
curl http://localhost:2600/jobs/abc123-def456-ghi789
```

#### **List All Jobs:**
```bash
curl http://localhost:2600/jobs
```

**Expected Response:**
```json
{
  "jobs": [
    {
      "job_id": "abc123-def456-ghi789",
      "status": "completed",
      "evaluation_mode": "paired",
      "backbone": "cnn14",
      "metrics_count": 9
    },
    {
      "job_id": "def456-ghi789-jkl012",
      "status": "running",
      "evaluation_mode": "unpaired"
    }
  ]
}
```

## 🐍 Programming Language Examples

### Python with Requests
```python
import requests
import json
import time

# API configuration
BASE_URL = "http://localhost:2600"

def check_api_health():
    """Check if API is running and healthy"""
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        health = response.json()
        print(f"✅ API Status: {health['status']}")
        print(f"🔧 CUDA Available: {health['cuda_available']}")
        print(f"📦 Models Preloaded: {health['models']['preloaded']}")
        return True
    else:
        print(f"❌ API Health Check Failed: {response.status_code}")
        return False

def submit_evaluation(generated_files, reference_files, **kwargs):
    """Submit audio files for evaluation"""
    
    # Prepare files
    files = {}
    for i, file_path in enumerate(generated_files):
        files[f'generated_files'] = files.get('generated_files', []) + [open(file_path, 'rb')]
    for i, file_path in enumerate(reference_files):
        files[f'reference_files'] = files.get('reference_files', []) + [open(file_path, 'rb')]
    
    # Prepare data
    data = {
        'backbone': kwargs.get('backbone', 'cnn14'),
        'sampling_rate': kwargs.get('sampling_rate', 16000),
        'metrics': kwargs.get('metrics', None),
        'limit_num': kwargs.get('limit_num', None)
    }
    
    # Submit evaluation
    response = requests.post(f"{BASE_URL}/evaluate", files=files, data=data)
    
    # Close file handles
    for file_list in files.values():
        for file_handle in file_list:
            file_handle.close()
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Evaluation submitted successfully!")
        print(f"🆔 Job ID: {result['job_id']}")
        print(f"📊 Status: {result['status']}")
        return result
    else:
        print(f"❌ Evaluation failed: {response.status_code}")
        print(f"Error: {response.text}")
        return None

def get_evaluation_results(job_id):
    """Get evaluation results by job ID"""
    response = requests.get(f"{BASE_URL}/jobs/{job_id}")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Failed to get results: {response.status_code}")
        return None

# Example usage
if __name__ == "__main__":
    # Check API health
    if not check_api_health():
        exit(1)
    
    # Submit evaluation
    generated_files = ["generated1.wav", "generated2.wav"]
    reference_files = ["reference1.wav", "reference2.wav"]
    
    result = submit_evaluation(
        generated_files, 
        reference_files,
        backbone="cnn14",
        sampling_rate=16000,
        metrics="FAD,ISc,FD"
    )
    
    if result:
        job_id = result['job_id']
        
        # Wait for completion and get results
        while True:
            results = get_evaluation_results(job_id)
            if results and results['status'] == 'completed':
                print("🎉 Evaluation completed!")
                print(f"📊 Metrics: {json.dumps(results['metrics'], indent=2)}")
                break
            elif results and results['status'] == 'failed':
                print(f"❌ Evaluation failed: {results.get('error', 'Unknown error')}")
                break
            else:
                print("⏳ Evaluation in progress...")
                time.sleep(5)
```

### Python with HTTPX (Async)
```python
import httpx
import asyncio
import json

async def evaluate_audio_async(generated_files, reference_files, **kwargs):
    """Async evaluation using HTTPX"""
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Prepare files
        files = []
        for file_path in generated_files:
            files.append(('generated_files', open(file_path, 'rb')))
        for file_path in reference_files:
            files.append(('reference_files', open(file_path, 'rb')))
        
        # Prepare data
        data = {
            'backbone': kwargs.get('backbone', 'cnn14'),
            'sampling_rate': kwargs.get('sampling_rate', 16000),
            'metrics': kwargs.get('metrics', None)
        }
        
        try:
            # Submit evaluation
            response = await client.post(
                "http://localhost:2600/evaluate",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Evaluation completed: {result['job_id']}")
                return result
            else:
                print(f"❌ Evaluation failed: {response.status_code}")
                return None
                
        finally:
            # Close file handles
            for _, file_handle in files:
                file_handle.close()

# Example usage
async def main():
    generated_files = ["generated1.wav", "generated2.wav"]
    reference_files = ["reference1.wav", "reference2.wav"]
    
    result = await evaluate_audio_async(
        generated_files,
        reference_files,
        backbone="cnn14",
        metrics="FAD,ISc"
    )
    
    if result:
        print(f"📊 Results: {json.dumps(result['metrics'], indent=2)}")

# Run async function
asyncio.run(main())
```

### JavaScript/Node.js
```javascript
const FormData = require('form-data');
const fs = require('fs');
const fetch = require('node-fetch');

class AudioLDMClient {
    constructor(baseUrl = 'http://localhost:2600') {
        this.baseUrl = baseUrl;
    }
    
    async checkHealth() {
        const response = await fetch(`${this.baseUrl}/health`);
        const health = await response.json();
        console.log(`✅ API Status: ${health.status}`);
        console.log(`🔧 CUDA Available: ${health.cuda_available}`);
        return health;
    }
    
    async submitEvaluation(generatedFiles, referenceFiles, options = {}) {
        const form = new FormData();
        
        // Add generated files
        generatedFiles.forEach(filePath => {
            form.append('generated_files', fs.createReadStream(filePath));
        });
        
        // Add reference files
        referenceFiles.forEach(filePath => {
            form.append('reference_files', fs.createReadStream(filePath));
        });
        
        // Add options
        if (options.backbone) form.append('backbone', options.backbone);
        if (options.samplingRate) form.append('sampling_rate', options.samplingRate);
        if (options.metrics) form.append('metrics', options.metrics);
        if (options.limitNum) form.append('limit_num', options.limitNum);
        
        const response = await fetch(`${this.baseUrl}/evaluate`, {
            method: 'POST',
            body: form
        });
        
        if (response.ok) {
            const result = await response.json();
            console.log(`✅ Evaluation submitted: ${result.job_id}`);
            return result;
        } else {
            console.error(`❌ Evaluation failed: ${response.status}`);
            return null;
        }
    }
    
    async getResults(jobId) {
        const response = await fetch(`${this.baseUrl}/jobs/${jobId}`);
        if (response.ok) {
            return await response.json();
        } else {
            console.error(`❌ Failed to get results: ${response.status}`);
            return null;
        }
    }
    
    async waitForCompletion(jobId, maxWaitTime = 300000) {
        const startTime = Date.now();
        
        while (Date.now() - startTime < maxWaitTime) {
            const results = await this.getResults(jobId);
            
            if (results.status === 'completed') {
                console.log('🎉 Evaluation completed!');
                return results;
            } else if (results.status === 'failed') {
                console.error(`❌ Evaluation failed: ${results.error}`);
                return null;
            } else {
                console.log('⏳ Evaluation in progress...');
                await new Promise(resolve => setTimeout(resolve, 5000));
            }
        }
        
        console.error('⏰ Evaluation timeout');
        return null;
    }
}

// Example usage
async function main() {
    const client = new AudioLDMClient();
    
    // Check health
    await client.checkHealth();
    
    // Submit evaluation
    const result = await client.submitEvaluation(
        ['generated1.wav', 'generated2.wav'],
        ['reference1.wav', 'reference2.wav'],
        {
            backbone: 'cnn14',
            samplingRate: 16000,
            metrics: 'FAD,ISc,FD'
        }
    );
    
    if (result) {
        // Wait for completion
        const finalResults = await client.waitForCompletion(result.job_id);
        if (finalResults) {
            console.log('📊 Metrics:', JSON.stringify(finalResults.metrics, null, 2));
        }
    }
}

main().catch(console.error);
```

### cURL Examples for Different Scenarios

#### **Quick Quality Check (Recommended Metrics Only)**
```bash
curl -X POST "http://localhost:2600/evaluate" \
  -F "generated_files=@my_generated_audio.wav" \
  -F "reference_files=@my_reference_audio.wav" \
  -F "metrics=FAD,ISc"
```

#### **Music Evaluation with MERT Backbone**
```bash
curl -X POST "http://localhost:2600/evaluate" \
  -F "generated_files=@music_gen1.wav" \
  -F "generated_files=@music_gen2.wav" \
  -F "reference_files=@music_ref1.wav" \
  -F "reference_files=@music_ref2.wav" \
  -F "backbone=mert" \
  -F "sampling_rate=32000" \
  -F "metrics=FAD,ISc,FD"
```

#### **High-Quality Speech Evaluation**
```bash
curl -X POST "http://localhost:2600/evaluate" \
  -F "generated_files=@speech_gen1.wav" \
  -F "generated_files=@speech_gen2.wav" \
  -F "reference_files=@speech_ref1.wav" \
  -F "reference_files=@speech_ref2.wav" \
  -F "backbone=cnn14" \
  -F "sampling_rate=16000" \
  -F "metrics=FAD,ISc,KL,PSNR,SSIM"
```

#### **Batch Processing with File Limit**
```bash
curl -X POST "http://localhost:2600/evaluate" \
  -F "generated_files=@batch_gen_*.wav" \
  -F "reference_files=@batch_ref_*.wav" \
  -F "backbone=cnn14" \
  -F "limit_num=50" \
  -F "metrics=FAD,ISc"
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

## 🔧 API Troubleshooting & Common Issues

### 🚨 Common Error Messages & Solutions

#### **1. Connection Errors**
```bash
# Error: Connection refused
curl: (7) Failed to connect to localhost port 2600: Connection refused
```
**Solutions:**
- Check if the API server is running: `docker ps` or `ps aux | grep uvicorn`
- Verify the correct port: `curl http://localhost:2600/health`
- Check firewall settings if using remote server
- Ensure Docker container is running: `docker run --gpus all -p 2600:2600 ghcr.io/bmwas/audiollmtest:latest`

#### **2. Model Loading Errors**
```json
{
  "status": "failed",
  "error": "Failed to load model cnn14_16000: CUDA out of memory"
}
```
**Solutions:**
- Reduce batch size or use CPU: Set `CUDA_VISIBLE_DEVICES=""` to force CPU mode
- Free GPU memory: `nvidia-smi` to check memory usage
- Use smaller audio files or reduce `limit_num` parameter
- Restart the container to clear GPU memory

#### **3. File Upload Errors**
```json
{
  "status": "failed", 
  "error": "No generated files provided"
}
```
**Solutions:**
- Ensure files are properly attached: `-F "generated_files=@file.wav"`
- Check file paths are correct and files exist
- Verify file permissions: `ls -la file.wav`
- Use absolute paths if relative paths fail

#### **4. Invalid Parameter Errors**
```json
{
  "status": "failed",
  "error": "Invalid backbone 'invalid_model'. Available: ['cnn14', 'mert']"
}
```
**Solutions:**
- Use valid backbone values: `"cnn14"` or `"mert"`
- Check sampling rate: Must be `16000` or `32000`
- Verify metrics spelling: `"FAD,ISc,FD"` not `"fad,isc,fd"`
- Use comma-separated values for metrics: `"FAD,ISc"` not `"FAD ISc"`

#### **5. Evaluation Mode Errors**
```json
{
  "status": "failed",
  "error": "KL metric requires paired evaluation mode"
}
```
**Solutions:**
- Use paired mode: Ensure same number of files with matching names
- Or use unpaired metrics only: `"FAD,ISc,FD,KID"`
- Check file naming: `generated1.wav` ↔ `reference1.wav`

### 🔍 Debugging Steps

#### **Step 1: Check API Health**
```bash
curl http://localhost:2600/health
```
**Expected:** `{"status": "healthy", "cuda_available": true, ...}`

#### **Step 2: Verify Model Status**
```bash
curl http://localhost:2600/models
```
**Expected:** `{"preloading_status": "completed", "model_count": 4}`

#### **Step 3: Test Simple Evaluation**
```bash
# Create test files first
python gen_test_file.py

# Test with minimal parameters
curl -X POST "http://localhost:2600/evaluate" \
  -F "generated_files=@example/paired/generated_000.wav" \
  -F "reference_files=@example/reference/reference_000.wav" \
  -F "metrics=FAD"
```

#### **Step 4: Check Logs**
```bash
# Docker logs
docker logs <container_id>

# Or check log file
tail -f /tmp/audioldm_api.log
```

### ⚡ Performance Optimization

#### **Slow Evaluation Times**
**Symptoms:** Evaluation takes >5 minutes for small files
**Solutions:**
- Use GPU: `docker run --gpus all ...`
- Preload models: Check `/models` endpoint shows `"preloaded": true`
- Reduce file size: Use shorter audio clips for testing
- Use recommended metrics only: `"FAD,ISc"` instead of all metrics

#### **High Memory Usage**
**Symptoms:** Out of memory errors, system slowdown
**Solutions:**
- Limit concurrent requests: Process files in smaller batches
- Use `limit_num` parameter to restrict evaluation size
- Monitor memory: `curl http://localhost:2600/health` shows memory stats
- Restart container periodically to clear memory

#### **Network Timeouts**
**Symptoms:** Connection timeouts during file upload
**Solutions:**
- Increase timeout: Use `--timeout 300` with curl
- Compress audio files: Use smaller file formats
- Upload in smaller batches: Process fewer files per request
- Use local server: Avoid network latency for large files

### 🐛 Advanced Debugging

#### **Enable Verbose Logging**
```bash
# Set environment variable for detailed logs
docker run --gpus all -p 2600:2600 \
  -e PYTHONUNBUFFERED=1 \
  -e LOG_LEVEL=DEBUG \
  ghcr.io/bmwas/audiollmtest:latest
```

#### **Check System Resources**
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor CPU and memory
htop

# Check disk space
df -h
```

#### **Test Individual Components**
```bash
# Test model loading
curl http://localhost:2600/models

# Test file upload (without evaluation)
curl -X POST "http://localhost:2600/evaluate" \
  -F "generated_files=@test.wav" \
  -F "reference_files=@test.wav" \
  -F "limit_num=1" \
  -F "metrics=FAD"
```

### 📊 Expected Behavior & Timing

#### **Normal Startup Sequence**
1. **Container Start**: 0-10 seconds
2. **Model Preloading**: 60-120 seconds (first time), 40-80 seconds (cached)
3. **API Ready**: Health check returns `"status": "healthy"`

#### **Normal Evaluation Times**
- **Small files** (< 10 seconds): 15-30 seconds
- **Medium files** (10-60 seconds): 30-60 seconds  
- **Large files** (> 60 seconds): 60-120 seconds
- **Batch processing** (10+ files): 2-5 minutes

#### **Expected Response Times**
- **Health check**: < 1 second
- **Model status**: < 1 second
- **Job submission**: < 5 seconds
- **Result retrieval**: < 1 second

### 🆘 Getting Help

#### **Check Logs First**
```bash
# Real-time log monitoring
docker logs -f <container_id>

# Or check persistent logs
tail -f /tmp/audioldm_api.log
```

#### **Common Log Patterns**
```
# Successful startup
INFO:app.models - 🎉 Model preloading completed!
INFO:app.models - 📊 Summary: 4 successful, 0 failed

# Evaluation in progress  
INFO:app.api - 🚀 Starting evaluation computation for job abc123...

# Error patterns
ERROR:app.api - ❌ Evaluation failed for job abc123: CUDA out of memory
WARNING:app.models - ⚠️ Model cnn14_16000 not preloaded, creating on-demand...
```

#### **Report Issues**
When reporting issues, include:
1. **Error message**: Full error response from API
2. **Request details**: curl command or code used
3. **System info**: `curl http://localhost:2600/health`
4. **Logs**: Relevant log entries
5. **File details**: Audio file format, size, duration

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
