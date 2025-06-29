#!/usr/bin/env python3

"""
Shared utilities for the Lunar Landslide Prototype
Common functions used across multiple processing scripts
"""

import subprocess
import sys
import shlex
import time
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import yaml
import os


def run_command(cmd: Union[str, List[str]], description: str = "", cwd: Optional[Path] = None) -> float:
    """Execute a command with timing and error handling (secure version)
    
    Args:
        cmd: Command to execute (string will be parsed safely, list preferred)
        description: Human-readable description of the command
        cwd: Working directory for command execution
    
    Returns:
        Elapsed execution time in seconds
    
    Raises:
        SystemExit: If command fails
    """
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    
    # Parse command safely if it's a string
    if isinstance(cmd, str):
        cmd_list = shlex.split(cmd)
        print(f"COMMAND: {' '.join(shlex.quote(arg) for arg in cmd_list)}")
    else:
        cmd_list = cmd
        print(f"COMMAND: {' '.join(shlex.quote(str(arg)) for arg in cmd_list)}")
    
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd_list,
            check=True,
            cwd=cwd,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ COMPLETED in {elapsed:.1f}s: {description}")
        return elapsed
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ FAILED after {elapsed:.1f}s: {description}")
        print(f"Error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ FAILED after {elapsed:.1f}s: {description}")
        print(f"Command not found: {e}")
        sys.exit(1)


def run_command_capture(cmd: List[str], description: str = "") -> subprocess.CompletedProcess:
    """Execute shell command with error handling and output capture (secure version)
    
    Args:
        cmd: Command arguments as list (secure)
        description: Human-readable description
    
    Returns:
        CompletedProcess result
    
    Raises:
        SystemExit: If command fails
    """
    print(f"Running: {description}")
    print(f"Command: {' '.join(shlex.quote(str(arg)) for arg in cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if result.stdout:
            print(f"Output: {result.stdout}")
        
        return result
        
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        print(f"Return code: {e.returncode}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Command not found: {e}")
        sys.exit(1)


def validate_file_exists(file_path: Union[str, Path], description: str = "") -> Path:
    """Validate that a file exists and return Path object
    
    Args:
        file_path: Path to validate
        description: Description for error messages
    
    Returns:
        Validated Path object
    
    Raises:
        SystemExit: If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        print(f"Error: {description or 'File'} not found: {path}")
        sys.exit(1)
    return path


def validate_files_exist(patterns: List[str], description: str = "") -> List[Path]:
    """Find and validate files matching glob patterns
    
    Args:
        patterns: List of glob patterns to search
        description: Description for error messages
    
    Returns:
        List of validated Path objects
    
    Raises:
        SystemExit: If no files found
    """
    import glob
    
    files = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        files.extend([Path(f) for f in matches])
    
    if not files:
        print(f"Error: No {description or 'files'} found matching patterns: {patterns}")
        sys.exit(1)
    
    return files


def safe_percentile(array, percentile, valid_mask=None):
    """Safely compute percentile with validation
    
    Args:
        array: Input array
        percentile: Percentile to compute (0-100)
        valid_mask: Boolean mask for valid values
    
    Returns:
        Percentile value or None if no valid data
    """
    import numpy as np
    
    if valid_mask is not None:
        valid_data = array[valid_mask]
    else:
        valid_data = array[~np.isnan(array)]
    
    if len(valid_data) == 0:
        print(f"Warning: No valid data for percentile computation")
        return None
    
    return np.percentile(valid_data, percentile)


def print_array_stats(array, name="Array"):
    """Print descriptive statistics for an array
    
    Args:
        array: NumPy array
        name: Name for display
    """
    import numpy as np
    
    valid_data = array[~np.isnan(array)]
    
    print(f"\n{name} Statistics:")
    print(f"  Shape: {array.shape}")
    print(f"  Valid pixels: {len(valid_data):,} / {array.size:,}")
    
    if len(valid_data) > 0:
        print(f"  Min: {valid_data.min():.4f}")
        print(f"  Max: {valid_data.max():.4f}")
        print(f"  Mean: {valid_data.mean():.4f}")
        print(f"  Std: {valid_data.std():.4f}")
        print(f"  P25: {np.percentile(valid_data, 25):.4f}")
        print(f"  P50: {np.percentile(valid_data, 50):.4f}")
        print(f"  P75: {np.percentile(valid_data, 75):.4f}")
        print(f"  P90: {np.percentile(valid_data, 90):.4f}")


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load configuration from YAML file
    
    Args:
        config_path: Path to config file (default: ../config.yaml)
    
    Returns:
        Configuration dictionary
    
    Raises:
        SystemExit: If config file not found or invalid
    """
    if config_path is None:
        # Default config path relative to this script
        config_path = Path(__file__).parent.parent / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Loaded configuration from: {config_path}")
        return config
        
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get nested configuration value using dot notation
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., "terrain.slope_threshold_degrees")
        default: Default value if key not found
    
    Returns:
        Configuration value or default
    
    Example:
        slope_thresh = get_config_value(config, "terrain.slope_threshold_degrees", 25)
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        if default is not None:
            print(f"Warning: Configuration key '{key_path}' not found, using default: {default}")
            return default
        else:
            print(f"Error: Required configuration key '{key_path}' not found and no default provided")
            sys.exit(1)


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration contains required keys
    
    Args:
        config: Configuration dictionary
    
    Returns:
        True if valid, exits on invalid
    """
    required_sections = [
        'photometric', 'terrain', 'texture', 'boulder', 
        'ml', 'data', 'performance', 'quality'
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in config:
            missing_sections.append(section)
    
    if missing_sections:
        print(f"Error: Missing required configuration sections: {missing_sections}")
        sys.exit(1)
    
    # Validate specific critical values
    critical_keys = [
        'terrain.slope_threshold_degrees',
        'terrain.curvature_threshold', 
        'texture.contrast_percentile',
        'ml.target_iou',
        'performance.max_processing_time_minutes'
    ]
    
    for key_path in critical_keys:
        value = get_config_value(config, key_path)
        if value is None:
            print(f"Error: Critical configuration value missing: {key_path}")
            sys.exit(1)
    
    print("Configuration validation passed")
    return True