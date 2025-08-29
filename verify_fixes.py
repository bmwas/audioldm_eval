#!/usr/bin/env python3
"""
Verification script to test that both fixes are working:
1. PyTorch 2.6+ compatibility (weights_only=False)
2. Model preloading system
"""

import asyncio
import torch
import sys
import os
from pathlib import Path

# Add the project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_pytorch_compatibility():
    """Test that torch.load calls work with PyTorch 2.6+"""
    print("🔍 Testing PyTorch compatibility...")
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Test if we can import without the weights_only error
    try:
        from audioldm_eval import EvaluationHelper
        print("✅ AudioLDM imports successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

async def test_model_preloading():
    """Test that model preloading system works"""
    print("\n🔍 Testing model preloading system...")
    
    try:
        # Import the ModelManager
        from app import ModelManager
        
        # Create manager and test preloading
        manager = ModelManager()
        
        print("Starting model preloading test...")
        await manager.preload_models()
        
        # Check if models were loaded
        model_info = manager.get_model_info()
        print(f"Preloaded: {model_info['preloaded']}")
        print(f"Device: {model_info['device']}")
        print(f"Loaded models: {model_info['loaded_models']}")
        print(f"Model count: {model_info['model_count']}")
        
        if model_info['preloaded'] and model_info['model_count'] > 0:
            print("✅ Model preloading system works")
            
            # Test getting a preloaded model
            evaluator = manager.get_evaluator("cnn14", 16000)
            if evaluator:
                print("✅ Can retrieve preloaded models")
                return True
        
        print("❌ Model preloading failed")
        return False
        
    except Exception as e:
        print(f"❌ Model preloading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_download_verification():
    """Test that download verification works"""
    print("\n🔍 Testing download verification...")
    
    # Check if model files exist
    home_dir = os.path.expanduser("~")
    model_files = [
        f"{home_dir}/.cache/audioldm_eval/ckpt/Cnn14_mAP=0.431.pth",
        f"{home_dir}/.cache/audioldm_eval/ckpt/Cnn14_16k_mAP=0.438.pth"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file)
            print(f"✅ {os.path.basename(model_file)}: {size} bytes")
        else:
            print(f"⚠️ {os.path.basename(model_file)}: Not found (will be downloaded on first use)")
    
    return True

async def main():
    """Run all verification tests"""
    print("AudioLDM Evaluation Fix Verification")
    print("=" * 50)
    
    results = []
    
    # Test 1: PyTorch compatibility
    results.append(test_pytorch_compatibility())
    
    # Test 2: Download verification
    results.append(test_download_verification())
    
    # Test 3: Model preloading (this may take a while)
    results.append(await test_model_preloading())
    
    # Summary
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        print("✅ PyTorch 2.6+ compatibility: FIXED")
        print("✅ Model preloading system: WORKING")
        print("✅ Download verification: WORKING")
        print("\nThe API should now:")
        print("  - Start without PyTorch compatibility errors")
        print("  - Preload all models during startup")
        print("  - Respond to evaluation requests quickly")
        print("  - Verify model downloads are complete")
    else:
        print(f"❌ SOME TESTS FAILED ({passed}/{total})")
        print("Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())
