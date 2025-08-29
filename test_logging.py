#!/usr/bin/env python3
"""
Test script to verify extensive logging is working properly
"""

import asyncio
import logging
import sys
import time
import json
from pathlib import Path

# Add the project to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure test logging to see all levels
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

logger = logging.getLogger(__name__)

def test_basic_logging():
    """Test basic logging functionality"""
    logger.info("🔍 Testing basic logging functionality...")
    
    try:
        from app import (
            logger as app_logger,
            perf_logger,
            download_logger,
            model_logger,
            api_logger,
            system_logger,
            get_system_stats
        )
        
        logger.info("✅ All specialized loggers imported successfully")
        
        # Test each logger
        app_logger.info("🧪 Testing main app logger")
        perf_logger.info("🧪 Testing performance logger")
        download_logger.info("🧪 Testing download logger")
        model_logger.info("🧪 Testing model logger")
        api_logger.info("🧪 Testing API logger")
        system_logger.info("🧪 Testing system logger")
        
        # Test system stats
        stats = get_system_stats()
        logger.info(f"📊 System stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic logging test failed: {e}")
        return False

async def test_model_manager_logging():
    """Test ModelManager logging"""
    logger.info("🔍 Testing ModelManager logging...")
    
    try:
        from app import ModelManager
        
        # Create a manager
        manager = ModelManager()
        logger.info("✅ ModelManager created successfully")
        
        # Test model info logging
        info = manager.get_model_info()
        logger.info(f"📊 Model info: {info}")
        
        # Test evaluator retrieval logging (this will fail gracefully since models aren't preloaded)
        try:
            evaluator = manager.get_evaluator("cnn14", 16000)
            logger.warning("⚠️ On-demand model creation worked (unexpected)")
        except Exception as e:
            logger.info(f"✅ Expected on-demand model creation handled gracefully: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ModelManager logging test failed: {e}")
        return False

def test_log_levels():
    """Test different log levels are working"""
    logger.info("🔍 Testing log levels...")
    
    try:
        from app import model_logger
        
        # Test all log levels
        model_logger.debug("🔍 DEBUG level message")
        model_logger.info("ℹ️ INFO level message")
        model_logger.warning("⚠️ WARNING level message")
        model_logger.error("❌ ERROR level message")
        
        logger.info("✅ All log levels tested successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Log levels test failed: {e}")
        return False

def test_performance_logging():
    """Test performance timing logging"""
    logger.info("🔍 Testing performance logging...")
    
    try:
        from app import perf_logger
        
        # Simulate a timed operation
        start_time = time.time()
        time.sleep(0.1)  # Simulate work
        end_time = time.time()
        duration = end_time - start_time
        
        perf_logger.info(f"🧪 Test operation completed in {duration:.3f}s")
        logger.info("✅ Performance logging test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance logging test failed: {e}")
        return False

def test_log_file_creation():
    """Test that log files are being created"""
    logger.info("🔍 Testing log file creation...")
    
    log_file = Path("/tmp/audioldm_api.log")
    
    try:
        # Generate some logs
        from app import logger as app_logger
        app_logger.info("🧪 Test message for log file")
        
        # Check if log file exists and has content
        if log_file.exists():
            with open(log_file, 'r') as f:
                content = f.read()
                if "Test message for log file" in content:
                    logger.info(f"✅ Log file created and contains expected content: {log_file}")
                    logger.info(f"📄 Log file size: {log_file.stat().st_size} bytes")
                    return True
                else:
                    logger.warning(f"⚠️ Log file exists but doesn't contain expected content")
                    return False
        else:
            logger.warning(f"⚠️ Log file not found: {log_file}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Log file test failed: {e}")
        return False

async def main():
    """Run all logging tests"""
    logger.info("AudioLDM Evaluation Logging Test")
    logger.info("=" * 50)
    
    tests = [
        ("Basic Logging", test_basic_logging),
        ("ModelManager Logging", test_model_manager_logging),
        ("Log Levels", test_log_levels),
        ("Performance Logging", test_performance_logging),
        ("Log File Creation", test_log_file_creation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running test: {test_name}")
        logger.info("-" * 30)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append(result)
            
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: CRASHED - {e}")
            results.append(False)
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("LOGGING TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        logger.info(f"🎉 ALL LOGGING TESTS PASSED ({passed}/{total})")
        logger.info("✅ Extensive logging system is fully functional")
        logger.info("\nLogging Features Verified:")
        logger.info("  📝 Multiple specialized loggers (API, Model, Performance, Downloads, System)")
        logger.info("  📊 System resource monitoring")
        logger.info("  📄 File logging to /tmp/audioldm_api.log")
        logger.info("  🔍 Debug, Info, Warning, Error levels")
        logger.info("  ⏱️ Performance timing measurements")
    else:
        logger.error(f"❌ SOME LOGGING TESTS FAILED ({passed}/{total})")
        logger.error("Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
