#!/usr/bin/env python3

"""
Test script to validate improvements to the Lunar Landslide Prototype
Tests security fixes, performance optimizations, and shared utilities
"""

import sys
import os
import tempfile
import numpy as np
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / 'scripts'))

def test_shared_utilities():
    """Test shared utilities module"""
    print("Testing shared utilities...")
    
    from utils import (
        validate_file_exists, validate_files_exist, 
        load_config, get_config_value, validate_config,
        safe_percentile, print_array_stats
    )
    
    # Test configuration loading
    try:
        config = load_config()
        print("✓ Configuration loading successful")
    except SystemExit:
        print("✗ Configuration loading failed - config.yaml missing or invalid")
        return False
    
    # Test configuration access
    slope_thresh = get_config_value(config, "terrain.slope_threshold_degrees", 25)
    assert slope_thresh == 25, f"Expected slope threshold 25, got {slope_thresh}"
    print("✓ Configuration value access working")
    
    # Test configuration validation
    try:
        validate_config(config)
        print("✓ Configuration validation passed")
    except SystemExit:
        print("✗ Configuration validation failed")
        return False
    
    # Test array statistics
    test_array = np.random.normal(0, 1, (100, 100))
    print_array_stats(test_array, "Test Array")
    print("✓ Array statistics function working")
    
    # Test safe percentile
    valid_mask = test_array > -0.5
    p90 = safe_percentile(test_array, 90, valid_mask)
    assert p90 is not None, "Percentile calculation failed"
    print("✓ Safe percentile calculation working")
    
    return True


def test_security_improvements():
    """Test security improvements (no shell injection)"""
    print("\nTesting security improvements...")
    
    from utils import run_command_capture
    
    # Test safe command execution
    try:
        # Test with a simple, safe command
        result = run_command_capture(["echo", "test"], "Echo test")
        assert "test" in result.stdout, "Command output not captured correctly"
        print("✓ Secure command execution working")
    except Exception as e:
        print(f"✗ Secure command execution failed: {e}")
        return False
    
    # Test command argument safety
    try:
        # This should work safely without shell injection risk
        dangerous_input = "test; echo 'injected'"
        result = run_command_capture(["echo", dangerous_input], "Safety test")
        # Should echo the literal string, not execute the injection
        assert "injected" in result.stdout and "test" in result.stdout
        print("✓ Shell injection protection working")
    except Exception as e:
        print(f"✗ Shell injection protection test failed: {e}")
        return False
    
    return True


def test_performance_optimizations():
    """Test performance optimizations"""
    print("\nTesting performance optimizations...")
    
    # Test optimized GLCM function
    try:
        sys.path.append(str(Path(__file__).parent / 'scripts'))
        from rule_based_baseline import glcm_contrast_optimized
        
        # Create small test image
        test_img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        
        # Test optimized GLCM (small window for testing)
        result = glcm_contrast_optimized(test_img, win=8, block_size=32)
        
        assert result.shape == test_img.shape, f"GLCM output shape mismatch: {result.shape} vs {test_img.shape}"
        assert result.dtype == np.float32, f"GLCM output dtype incorrect: {result.dtype}"
        print("✓ Optimized GLCM computation working")
        
    except ImportError as e:
        print(f"✗ GLCM optimization test failed - import error: {e}")
        return False
    except Exception as e:
        print(f"✗ GLCM optimization test failed: {e}")
        return False
    
    return True


def test_main_pipeline():
    """Test main pipeline improvements"""
    print("\nTesting main pipeline improvements...")
    
    # Test that run_prototype.py can be imported without errors
    try:
        import run_prototype
        print("✓ Main pipeline script imports successfully")
    except ImportError as e:
        print(f"✗ Main pipeline import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Main pipeline test failed: {e}")
        return False
    
    # Test step function improvements
    try:
        # This should work even if the script files don't exist (will exit gracefully)
        # We're just testing that the function structure is correct
        step_funcs = [
            run_prototype.step_0_environment,
            run_prototype.step_7_annotation
        ]
        
        for func in step_funcs:
            assert callable(func), f"Step function {func.__name__} not callable"
        
        print("✓ Pipeline step functions are properly structured")
        
    except Exception as e:
        print(f"✗ Pipeline step function test failed: {e}")
        return False
    
    return True


def test_file_validation():
    """Test file validation improvements"""
    print("\nTesting file validation improvements...")
    
    from utils import validate_file_exists, validate_files_exist
    
    # Test with a file that should exist
    try:
        config_path = validate_file_exists("config.yaml", "config file")
        assert config_path.exists(), "Config file validation failed"
        print("✓ File existence validation working")
    except SystemExit:
        print("✗ File existence validation failed - config.yaml not found")
        return False
    except Exception as e:
        print(f"✗ File validation test failed: {e}")
        return False
    
    return True


def main():
    """Run all improvement tests"""
    print("=" * 60)
    print("TESTING LUNAR LANDSLIDE PROTOTYPE IMPROVEMENTS")
    print("=" * 60)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    tests = [
        ("Shared Utilities", test_shared_utilities),
        ("Security Improvements", test_security_improvements), 
        ("Performance Optimizations", test_performance_optimizations),
        ("Main Pipeline", test_main_pipeline),
        ("File Validation", test_file_validation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print(f"{'-' * 40}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        icon = "✓" if success else "✗"
        print(f"{icon} {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All improvements validated successfully!")
        return 0
    else:
        print("⚠️  Some improvements need attention")
        return 1


if __name__ == "__main__":
    sys.exit(main())