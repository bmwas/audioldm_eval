#!/bin/bash
# Simple script to run the API test

echo "AudioLDM Evaluation API Test Runner"
echo "===================================="

# Check if API is running
echo "Checking if API is running on port 2600..."
if curl -s http://localhost:2600/health > /dev/null; then
    echo "✓ API is running"
else
    echo "✗ API is not running. Please start the API first:"
    echo "  docker run --gpus all -p 2600:2600 ghcr.io/bmwas/audiollmtest:latest"
    echo "  or"
    echo "  python app.py"
    exit 1
fi

# Install test dependencies if needed
if ! python -c "import requests" 2>/dev/null; then
    echo "Installing test dependencies..."
    pip install requests soundfile numpy
fi

# Run the test
echo ""
echo "Running API tests..."
python test_api.py "$@"
