#!/bin/bash
# Simple script to run AudioLDM evaluation tests

echo "AudioLDM Evaluation Test Runner"
echo "==============================="

# Parse arguments
TEST_TYPE="direct"
if [ "$1" = "api" ] || [ "$1" = "web" ]; then
    TEST_TYPE="api"
    shift
fi

if [ "$TEST_TYPE" = "api" ]; then
    echo "Running Web API tests..."
    
    # Check if API is running
    echo "Checking if API is running on port 2600..."
    API_URL="http://10.4.11.192:2600"
    if curl -s "$API_URL/health" > /dev/null; then
        echo "✓ API is running at $API_URL"
    else
        echo "✗ API is not running. Please start the API first:"
        echo "  docker run --gpus all -p 2600:2600 ghcr.io/bmwas/audiollmtest:latest"
        echo "  or"
        echo "  python app.py"
        exit 1
    fi

    # Install test dependencies if needed
    if ! python -c "import requests" 2>/dev/null; then
        echo "Installing web test dependencies..."
        pip install requests soundfile numpy
    fi

    # Run the web API test
    echo ""
    echo "Running Web API tests..."
    python test_web_api.py "$@"
    
else
    echo "Running direct library tests..."
    
    # Install test dependencies if needed
    if ! python -c "import soundfile" 2>/dev/null; then
        echo "Installing test dependencies..."
        pip install soundfile numpy
    fi

    # Run the direct test
    echo ""
    echo "Running direct library tests..."
    python test_api.py "$@"
fi

echo ""
echo "Usage:"
echo "  $0          # Run direct library test"
echo "  $0 api      # Run web API test"
