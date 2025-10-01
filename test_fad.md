# FAD-Only API Call Instructions

## Overview
Instructions for making an API call to the AudioLDM Evaluation API to get **only the FAD (Frechet Audio Distance)** metric.

## API Endpoint
```
POST http://localhost:2600/evaluate
```

## Required Parameters

### Files (multipart/form-data)
- `generated_files`: Generated audio file(s) to evaluate
- `reference_files`: Reference audio file(s) for comparison

### Form Data
- `backbone`: "cnn14" (recommended) or "mert"
- `sampling_rate`: 16000 or 32000
- `metrics`: "FAD" (to get only FAD metric)

## Example cURL Command

```bash
curl -X POST "http://localhost:2600/evaluate" \
  -F "generated_files=@predicted_audio_1.wav" \
  -F "reference_files=@ref_audio_1.wav" \
  -F "backbone=cnn14" \
  -F "sampling_rate=16000" \
  -F "metrics=FAD"
```

## Python Example

```python
import requests

# API configuration
api_url = "http://localhost:2600"

# Prepare files
files = {
    'generated_files': open('predicted_audio_1.wav', 'rb'),
    'reference_files': open('ref_audio_1.wav', 'rb')
}

# Prepare form data
data = {
    'backbone': 'cnn14',
    'sampling_rate': 16000,
    'metrics': 'FAD'
}

# Make API call
response = requests.post(f"{api_url}/evaluate", files=files, data=data)

# Close file handles
for file_handle in files.values():
    file_handle.close()

# Parse response
if response.status_code == 200:
    result = response.json()
    fad_score = result['metrics']['frechet_audio_distance']
    print(f"FAD Score: {fad_score}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

## Expected Response

```json
{
  "job_id": "abc123-def456-ghi789",
  "status": "completed",
  "metrics": {
    "frechet_audio_distance": 28.984039
  },
  "evaluation_mode": "unpaired"
}
```

## FAD Score Interpretation

- **Lower is better**: FAD measures the distance between generated and reference audio distributions
- **Typical ranges**:
  - < 1.0: Excellent quality
  - 1.0-3.0: Good quality  
  - 3.0-10.0: Moderate quality
  - > 10.0: Poor quality

## Notes

1. **File formats**: Supports WAV, MP3, FLAC, M4A
2. **Paired vs Unpaired**: FAD works in both modes
3. **Single file**: FAD can be calculated with just one generated and one reference file
4. **Timeout**: API call may take 30-60 seconds depending on file size
5. **Error handling**: Check response status code and handle errors appropriately

## Quick Test

Use the provided test script with FAD-only metrics:

```bash
python test_api.py --generated predicted_audio_1.wav --reference ref_audio_1.wav --metrics FAD
```
