# 📝 AudioLDM Evaluation API - Extensive Logging Documentation

## Overview

This document describes the comprehensive logging system implemented to track all changes and operations in the AudioLDM Evaluation API. The logging system provides detailed monitoring, debugging capabilities, and performance insights.

## 🏗️ Logging Architecture

### Multiple Specialized Loggers

The system implements 6 specialized loggers for different components:

1. **Main Logger** (`logger`) - General application events
2. **Performance Logger** (`perf_logger`) - Timing and performance metrics
3. **Download Logger** (`download_logger`) - Model download operations
4. **Model Logger** (`model_logger`) - Model loading and management
5. **API Logger** (`api_logger`) - API request/response tracking
6. **System Logger** (`system_logger`) - System resource monitoring

### Configuration

```python
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('/tmp/audioldm_api.log', mode='a')  # File output
    ]
)
```

## 🔍 Detailed Logging Features

### 1. System Resource Monitoring

**Function**: `get_system_stats()`
**Purpose**: Track CPU, memory, and disk usage
**Usage**: Called before/after major operations

```python
def get_system_stats():
    return {
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'memory_mb': process.memory_info().rss / 1024 / 1024,
        'memory_percent': process.memory_percent(),
        'disk_usage_gb': psutil.disk_usage('/').used / 1024 / 1024 / 1024
    }
```

### 2. Model Preloading Logging

**Logger**: `model_logger`
**Covers**: Model downloads, loading, caching, and retrieval

#### Startup Sequence Logging:
- 🚀 Preloading initiation
- 📦 Individual model loading (with progress indicators)
- ⏱️ Load times for each model configuration
- 📊 Success/failure statistics
- 💾 Cache status and model information
- 🎉 Completion summary

#### Sample Output:
```
2025-08-29 15:30:45,123 - app.models - INFO - 🚀 Starting model preloading...
2025-08-29 15:30:45,124 - app.models - INFO - CUDA device detected: NVIDIA RTX 4090
2025-08-29 15:30:45,125 - app.models - INFO - 📦 [1/4] Loading cnn14 model (sr=16000)...
2025-08-29 15:30:45,126 - app.models - INFO - 🔄 Initializing EvaluationHelper for cnn14...
2025-08-29 15:31:05,234 - app.models - INFO - ✅ cnn14 model (sr=16000) loaded and verified in 20.11s
2025-08-29 15:31:25,567 - app.models - INFO - 🎉 Model preloading completed!
2025-08-29 15:31:25,568 - app.models - INFO - 📊 Summary: 4 successful, 0 failed
2025-08-29 15:31:25,569 - app.models - INFO - ⏱️ Total preload time: 40.45s
```

### 3. Model Download Logging

**Logger**: `download_logger`
**Covers**: CNN14 model downloads with verification

#### Download Process Logging:
- 🔍 File existence checks
- ⬇️ Download initiation with URLs
- ⏱️ Download timing
- 📊 File size verification
- ✅ Success confirmation
- ❌ Error handling and retry logic

#### Sample Output:
```
2025-08-29 15:30:50,456 - panns.models.downloads - INFO - 📦 CNN14 model files missing, starting download process...
2025-08-29 15:30:50,457 - panns.models.downloads - INFO - ⬇️ Downloading Cnn14_mAP=0.431.pth (32kHz model)...
2025-08-29 15:31:15,789 - panns.models.downloads - INFO - ✅ Successfully downloaded Cnn14_mAP=0.431.pth in 25.33s (285.4MB)
```

### 4. API Request Logging

**Logger**: `api_logger`
**Covers**: Complete API request lifecycle

#### Request Processing Logging:
- 🚀 New request initiation with job ID
- 📝 Request parameters validation
- 📊 File upload processing
- 🎯 Evaluation mode determination
- 🔄 Model retrieval
- 💾 File saving with size tracking
- ✅ Success/failure outcomes

#### Sample Output:
```
2025-08-29 15:45:12,234 - app.api - INFO - 🚀 NEW EVALUATION REQUEST - Job ID: a1b2c3d4-e5f6-7890-abcd-ef1234567890
2025-08-29 15:45:12,235 - app.api - INFO - 📝 Request params: backbone=cnn14, sr=16000, limit=None
2025-08-29 15:45:12,236 - app.api - DEBUG - 📊 Files received: 3 generated, 3 reference
2025-08-29 15:45:12,456 - app.api - INFO - ✅ File upload completed for job a1b2c3d4 in 0.22s
2025-08-29 15:45:12,457 - app.api - INFO - 📊 File sizes - Generated: 15.2MB, Reference: 14.8MB
2025-08-29 15:45:12,458 - app.api - INFO - 🎯 Evaluation mode selected: paired
```

### 5. Performance Logging

**Logger**: `perf_logger`
**Covers**: Timing measurements for optimization

#### Performance Metrics:
- ⏱️ Model loading times
- 📊 Cache hit/miss performance
- 🔄 File upload durations
- 💻 Evaluation computation times
- 📈 System resource usage trends

#### Sample Output:
```
2025-08-29 15:45:13,123 - app.performance - DEBUG - Cache hit for cnn14_16000: 0.0001s
2025-08-29 15:45:13,124 - app.performance - INFO - File upload for job a1b2c3d4: 0.22s, 30.0MB total
2025-08-29 15:45:45,678 - app.performance - INFO - Evaluation computation for job a1b2c3d4: 32.55s
```

### 6. Error and Exception Logging

**All Loggers**: Comprehensive error tracking

#### Error Handling Features:
- ❌ Detailed error messages with context
- 🔍 Stack traces for debugging
- ⚠️ Warning messages for potential issues
- 🔄 Recovery attempt logging
- 📊 Error statistics tracking

## 📄 Log File Output

### File Location: `/tmp/audioldm_api.log`

All logging output is simultaneously sent to:
1. **Console** - Real-time monitoring
2. **Log File** - Persistent storage for analysis

### Log Rotation
- **Manual Management**: Users should implement log rotation as needed
- **Recommended**: Use `logrotate` for production deployments

## 🎯 Log Levels and Usage

### DEBUG Level
- 🔍 Detailed system information
- 📊 Resource usage statistics
- 🔧 Internal state changes
- 🧪 Development debugging info

### INFO Level
- ℹ️ Normal operation events
- ✅ Successful operations
- 📈 Progress indicators
- 🎯 Important state changes

### WARNING Level
- ⚠️ Potential issues
- 🔄 Fallback operations
- 📊 Performance concerns
- 🚨 Non-critical errors

### ERROR Level
- ❌ Operation failures
- 🚫 Critical errors
- 💥 Exception handling
- 🆘 System issues

## 🧪 Testing the Logging System

### Test Script: `test_logging.py`

Run the comprehensive logging test:
```bash
python test_logging.py
```

The test verifies:
- ✅ All logger initialization
- 📊 System stats collection
- 📄 Log file creation
- ⏱️ Performance measurements
- 🔍 All log levels functioning

## 🔧 Production Recommendations

### 1. Log Level Configuration
```python
# Production: INFO level to reduce noise
logging.getLogger('app').setLevel(logging.INFO)

# Development: DEBUG level for detailed insights
logging.getLogger('app').setLevel(logging.DEBUG)
```

### 2. Log File Management
```bash
# Set up log rotation
sudo apt-get install logrotate

# Create logrotate config
echo "/tmp/audioldm_api.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}" | sudo tee /etc/logrotate.d/audioldm-api
```

### 3. Monitoring and Alerting
- Monitor log file growth
- Set up alerts for ERROR level messages
- Track performance metrics trends
- Monitor system resource usage

## 📊 Example Complete Log Flow

Here's what a complete evaluation request looks like in the logs:

```
# Request Initiation
2025-08-29 15:45:12,234 - app.api - INFO - 🚀 NEW EVALUATION REQUEST - Job ID: abc123
2025-08-29 15:45:12,235 - app.system - DEBUG - Pre-evaluation system stats: {'cpu_percent': 15.2, 'memory_mb': 2048.5}

# File Processing
2025-08-29 15:45:12,456 - app.api - INFO - ✅ File upload completed in 0.22s
2025-08-29 15:45:12,457 - app.api - INFO - 📊 File sizes - Generated: 15.2MB, Reference: 14.8MB

# Model Retrieval
2025-08-29 15:45:12,500 - app.models - DEBUG - ✅ Retrieved preloaded model cnn14_16000 in 0.0001s
2025-08-29 15:45:12,501 - app.performance - DEBUG - Cache hit for cnn14_16000: 0.0001s

# Evaluation
2025-08-29 15:45:13,000 - app.api - INFO - 🚀 Starting evaluation computation...
2025-08-29 15:45:45,555 - app.api - INFO - ✅ Evaluation computation completed in 32.55s
2025-08-29 15:45:45,556 - app.performance - INFO - Evaluation computation: 32.55s

# Completion
2025-08-29 15:45:45,600 - app.system - DEBUG - Post-evaluation system stats: {'cpu_percent': 8.7, 'memory_mb': 2156.8}
```

## 🎉 Benefits of This Logging System

1. **🔍 Complete Visibility** - Track every operation from start to finish
2. **⏱️ Performance Optimization** - Identify bottlenecks and optimize
3. **🐛 Easy Debugging** - Detailed context for troubleshooting
4. **📊 Monitoring** - System health and resource usage tracking
5. **📈 Analytics** - Usage patterns and performance trends
6. **🚨 Alerting** - Quick identification of issues
7. **📝 Audit Trail** - Complete record of all operations

This extensive logging system ensures that every aspect of the AudioLDM Evaluation API is tracked, monitored, and debuggable, providing the foundation for reliable production operations and continuous improvement.
