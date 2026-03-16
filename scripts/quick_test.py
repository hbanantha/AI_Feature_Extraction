"""
Quick Test Script
==================
Run this to verify the installation and test basic functionality.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    modules = [
        ("numpy", "np"),
        ("torch", None),
        ("torchvision", None),
        ("segmentation_models_pytorch", "smp"),
        ("rasterio", None),
        ("cv2", None),
        ("albumentations", "A"),
        ("geopandas", "gpd"),
        ("tqdm", None),
        ("yaml", None),
    ]

    all_ok = True
    for module_name, alias in modules:
        try:
            if alias:
                exec(f"import {module_name} as {alias}")
            else:
                exec(f"import {module_name}")
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ {module_name}: {e}")
            all_ok = False

    return all_ok


def test_model_creation():
    """Test that models can be created."""
    print("\nTesting model creation...")

    try:
        import torch
        from src.models import create_model

        config = {
            "model": {
                "segmentation": {
                    "encoder_name": "mobilenetv2_100",
                    "encoder_weights": None,  # No pretrained weights for test
                    "decoder": "unet",
                    "in_channels": 3,
                    "classes": 7
                }
            }
        }

        model = create_model(config)

        # Test forward pass
        x = torch.randn(1, 3, 256, 256)
        y = model(x)

        print(f"  ✓ Model created successfully")
        print(f"  ✓ Input shape: {x.shape}")
        print(f"  ✓ Output shape: {y.shape}")

        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Parameters: {params:,}")

        return True

    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        return False


def test_tiling():
    """Test the tiling pipeline."""
    print("\nTesting tiling pipeline...")

    try:
        from src.preprocessing import GeoTIFFTiler

        tiler = GeoTIFFTiler(
            tile_size=256,
            overlap=32,
            min_valid_ratio=0.7
        )

        print(f"  ✓ Tiler created")
        print(f"  ✓ Tile size: {tiler.tile_size}")
        print(f"  ✓ Overlap: {tiler.overlap}")

        return True

    except Exception as e:
        print(f"  ✗ Tiling test failed: {e}")
        return False


def test_data_loader():
    """Test data loader creation."""
    print("\nTesting data loader...")

    try:
        from src.preprocessing import get_training_augmentation

        config = {"augmentation": {"train": {}}}
        transform = get_training_augmentation(config)

        print(f"  ✓ Augmentation pipeline created")
        print(f"  ✓ Transforms: {len(transform.transforms)}")

        return True

    except Exception as e:
        print(f"  ✗ Data loader test failed: {e}")
        return False


def test_sample_data_generation():
    """Test synthetic data generation."""
    print("\nTesting sample data generation...")

    try:
        from scripts.generate_sample_data import generate_synthetic_tile

        image, mask = generate_synthetic_tile(
            size=256,
            num_buildings=3,
            has_road=True,
            has_water=True
        )

        print(f"  ✓ Synthetic tile generated")
        print(f"  ✓ Image shape: {image.shape}")
        print(f"  ✓ Mask shape: {mask.shape}")
        print(f"  ✓ Unique classes in mask: {set(mask.flatten())}")

        return True

    except Exception as e:
        print(f"  ✗ Sample data generation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("AI Feature Extraction - Quick Test")
    print("=" * 50)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Tiling Pipeline", test_tiling()))
    results.append(("Data Loader", test_data_loader()))
    results.append(("Sample Data", test_sample_data_generation()))

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 50)

    if all_passed:
        print("\nAll tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("  1. Generate sample data: python scripts/generate_sample_data.py --type dataset --output data")
        print("  2. Train model: python main.py train --config configs/config.yaml")
    else:
        print("\nSome tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())