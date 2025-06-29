#!/usr/bin/env python3

"""
Lunar Landslide Prototype - Main CLI Entry Point
Orchestrates the complete pipeline from data acquisition to final results
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional, Union

# Import shared utilities
sys.path.append(str(Path(__file__).parent / 'scripts'))
from utils import run_command, validate_file_exists


def step_0_environment():
    """Step 0: Environment setup"""
    scripts_dir = Path(__file__).parent / "scripts" / "00_env_setup"
    
    print("\nEnvironment setup files available:")
    print(f"- {scripts_dir}/environment.yml")
    print(f"- {scripts_dir}/gdal_bashrc_snippet.sh")
    print("\nTo set up the environment, run:")
    print("conda env create -f scripts/00_env_setup/environment.yml")
    print("conda activate moonai")
    print("source scripts/00_env_setup/gdal_bashrc_snippet.sh")
    
    return 0

def step_1_data_acquisition() -> float:
    """Step 1: Data acquisition"""
    script_path = validate_file_exists(
        Path(__file__).parent / "scripts" / "01_data_acquisition.sh",
        "Data acquisition script"
    )
    cmd = ["bash", str(script_path)]
    return run_command(cmd, "Data Acquisition")

def step_2_preprocessing() -> float:
    """Step 2: Raster preprocessing"""
    script_path = validate_file_exists(
        Path(__file__).parent / "scripts" / "02_raster_preprocessing.py",
        "Raster preprocessing script"
    )
    cmd = ["python", str(script_path)]
    return run_command(cmd, "Raster Preprocessing")

def step_3_coregistration() -> float:
    """Step 3: Co-registration"""
    script_path = validate_file_exists(
        Path(__file__).parent / "scripts" / "03_coregistration.sh",
        "Co-registration script"
    )
    cmd = ["bash", str(script_path)]
    return run_command(cmd, "Co-registration (requires manual GCP collection)")

def step_4_hapke() -> float:
    """Step 4: Hapke normalization"""
    script_path = validate_file_exists(
        Path(__file__).parent / "scripts" / "04_hapke_normalisation.py",
        "Hapke normalization script"
    )
    cmd = ["python", str(script_path)]
    return run_command(cmd, "Hapke Normalization")

def step_5_terrain() -> float:
    """Step 5: Terrain derivatives"""
    script_path = validate_file_exists(
        Path(__file__).parent / "scripts" / "05_terrain_derivatives.py",
        "Terrain derivatives script"
    )
    cmd = ["python", str(script_path)]
    return run_command(cmd, "Terrain Derivatives")

def step_6_baseline() -> float:
    """Step 6: Rule-based baseline"""
    script_path = validate_file_exists(
        Path(__file__).parent / "scripts" / "06_rule_based_baseline.py",
        "Rule-based baseline script"
    )
    cmd = ["python", str(script_path)]
    return run_command(cmd, "Rule-based Baseline")

def step_7_annotation() -> float:
    """Step 7: Annotation sprint (manual)"""
    print("\n" + "="*60)
    print("STEP 7: Annotation Sprint (Manual)")
    print("="*60)
    print("This step requires manual annotation work:")
    print("- 30 landslide polygons in QGIS")
    print("- 300 boulder annotations in LabelMe")
    print("- Dataset split (70% train, 15% val, 15% test)")
    
    annotation_guide = Path(__file__).parent / "scripts" / "07_annotation_sprint.md"
    if annotation_guide.exists():
        print(f"\nSee: {annotation_guide} for detailed instructions")
    else:
        print("\nWarning: Annotation guide not found at expected location")
    
    return 0

def step_8_ml_models() -> float:
    """Step 8: Light ML models"""
    script_path = validate_file_exists(
        Path(__file__).parent / "scripts" / "08_light_ml_models.py",
        "ML models training script"
    )
    cmd = ["python", str(script_path)]
    return run_command(cmd, "Light ML Models Training")

def step_9_fusion() -> float:
    """Step 9: Fusion and filtering"""
    script_path = validate_file_exists(
        Path(__file__).parent / "scripts" / "09_fusion_and_filter.py",
        "Fusion and filtering script"
    )
    cmd = ["python", str(script_path)]
    return run_command(cmd, "Cross-scale Fusion & Physics Filter")

def step_10_metrics() -> float:
    """Step 10: Metrics audit"""
    script_path = validate_file_exists(
        Path(__file__).parent / "scripts" / "10_metrics_audit.py",
        "Metrics audit script"
    )
    cmd = ["python", str(script_path)]
    return run_command(cmd, "Metrics & Runtime Audit")

def step_11_visuals() -> float:
    """Step 11: Visuals and packaging"""
    script_path = validate_file_exists(
        Path(__file__).parent / "scripts" / "11_visuals_packaging.py",
        "Visuals and packaging script"
    )
    cmd = ["python", str(script_path)]
    return run_command(cmd, "Visuals & Packaging")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Lunar Landslide Prototype Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_prototype.py --step 1          # Run data acquisition only
  python run_prototype.py --step 2-5        # Run steps 2 through 5
  python run_prototype.py --all             # Run complete pipeline
  python run_prototype.py --aoi data/aoi.geojson --all  # Run with specific AOI
        """
    )
    
    parser.add_argument(
        '--step', 
        type=str,
        help='Step(s) to run (e.g., "1", "2-5", "all")'
    )
    
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Run all steps in sequence'
    )
    
    parser.add_argument(
        '--aoi',
        type=str,
        default='data/aoi.geojson',
        help='Path to AOI GeoJSON file (default: data/aoi.geojson)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available steps'
    )
    
    args = parser.parse_args()
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Define step functions
    steps = {
        0: ("Environment Setup", step_0_environment),
        1: ("Data Acquisition", step_1_data_acquisition),
        2: ("Raster Preprocessing", step_2_preprocessing),
        3: ("Co-registration", step_3_coregistration),
        4: ("Hapke Normalization", step_4_hapke),
        5: ("Terrain Derivatives", step_5_terrain),
        6: ("Rule-based Baseline", step_6_baseline),
        7: ("Annotation Sprint", step_7_annotation),
        8: ("Light ML Models", step_8_ml_models),
        9: ("Fusion & Filter", step_9_fusion),
        10: ("Metrics Audit", step_10_metrics),
        11: ("Visuals & Packaging", step_11_visuals),
    }
    
    if args.list:
        print("Available pipeline steps:")
        for step_num, (step_name, _) in steps.items():
            print(f"  {step_num}: {step_name}")
        return
    
    if not args.step and not args.all:
        parser.print_help()
        return
    
    # Validate AOI file
    if not Path(args.aoi).exists():
        print(f"Error: AOI file not found: {args.aoi}")
        sys.exit(1)
    
    print(f"Lunar Landslide Prototype Pipeline")
    print(f"AOI: {args.aoi}")
    print(f"Working directory: {project_dir}")
    
    # Determine which steps to run
    if args.all:
        steps_to_run = list(range(1, 12))  # Steps 1-11
    else:
        if '-' in args.step:
            # Range of steps
            start, end = map(int, args.step.split('-'))
            steps_to_run = list(range(start, end + 1))
        else:
            # Single step
            steps_to_run = [int(args.step)]
    
    print(f"Steps to run: {steps_to_run}")
    
    # Execute steps
    total_time = 0
    successful_steps = []
    
    for step_num in steps_to_run:
        if step_num not in steps:
            print(f"Error: Invalid step number: {step_num}")
            continue
        
        step_name, step_func = steps[step_num]
        
        try:
            elapsed = step_func()
            if elapsed is not None:
                total_time += elapsed
            successful_steps.append((step_num, step_name))
            
        except KeyboardInterrupt:
            print(f"\nPipeline interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"Error in step {step_num} ({step_name}): {e}")
            sys.exit(1)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully completed {len(successful_steps)} steps:")
    
    for step_num, step_name in successful_steps:
        print(f"  ✓ Step {step_num}: {step_name}")
    
    if total_time > 0:
        print(f"\nTotal execution time: {total_time:.1f}s ({total_time/60:.1f}m)")
    
    print(f"\nOutput files in: outputs/")
    print(f"Reports in: reports/")
    print(f"Notebook: notebooks/Prototype_Report.ipynb")

if __name__ == "__main__":
    main()