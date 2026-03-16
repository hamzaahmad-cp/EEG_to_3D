#!/usr/bin/env python3
"""
Verify that the project structure is set up correctly.
Checks imports, config, data files, and checkpoints.

Usage:
    python scripts/verify_setup.py
"""

import sys, os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}!{RESET} {msg}")

errors = 0

print("=" * 50)
print("  EEG-to-3D Project Setup Verification")
print("=" * 50)

# 1. Directory structure
print("\n[1] Directory structure")
required_dirs = [
    "src", "src/models", "src/data", "src/pipeline", "src/evaluation", "src/training",
    "config", "checkpoints", "data",
]
for d in required_dirs:
    if Path(d).exists():
        ok(d)
    else:
        fail(f"{d} — MISSING")
        errors += 1

# 2. Config
print("\n[2] Configuration")
config_path = Path("config/config.yaml")
if config_path.exists():
    ok("config/config.yaml exists")
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    required_keys = ['eeg_classification', 'contrastive_training', 'generation', 'class_prompts', 'data', 'checkpoints']
    for k in required_keys:
        if k in config:
            ok(f"  config['{k}']")
        else:
            fail(f"  config['{k}'] — MISSING")
            errors += 1
else:
    fail("config/config.yaml — MISSING")
    errors += 1

# 3. Source module imports
print("\n[3] Module imports")
import_tests = [
    ("src.models.eeg_classifier", "EEGClassifier"),
    ("src.models.contrastive_encoder", "ContrastiveEncoder"),
    ("src.data.data_loader", "CATVisDataLoader"),
    ("src.data.preprocessor", "DataPreprocessor"),
    ("src.pipeline.retrieval", "TextRetrieval"),
]
for module, cls in import_tests:
    try:
        mod = __import__(module, fromlist=[cls])
        getattr(mod, cls)
        ok(f"from {module} import {cls}")
    except Exception as e:
        fail(f"from {module} import {cls} — {e}")
        errors += 1

# 4. Data files
print("\n[4] Data files")
data_dir = Path(config['data']['root_dir']) if config_path.exists() else Path("data")
data_files = [
    config.get('data', {}).get('eeg_file', 'eeg_55_95_std.pth'),
    config.get('data', {}).get('splits_file', 'block_splits_by_image_all.pth'),
    config.get('data', {}).get('captions_file', 'captions_with_bbox_data.pth'),
    config.get('data', {}).get('imagenet_labels', 'imagenet_class_labels.txt'),
]
for f in data_files:
    p = data_dir / f
    if p.exists():
        ok(f"data/{f}")
    else:
        warn(f"data/{f} — NOT FOUND (needed for training/inference)")

images_dir = data_dir / config.get('data', {}).get('imagenet_images', 'imageNet_images')
if images_dir.exists():
    n = len(list(images_dir.iterdir()))
    ok(f"ImageNet images directory ({n} items)")
else:
    warn(f"{images_dir} — NOT FOUND (needed for GT comparisons)")

# 5. Checkpoints
print("\n[5] Checkpoints")
ckpt_dir = Path(config.get('checkpoints', {}).get('root_dir', 'checkpoints'))
for name in ['eeg_classifier', 'contrastive_model']:
    fname = config.get('checkpoints', {}).get(name, f'{name}_best.pth')
    p = ckpt_dir / fname
    if p.exists():
        size_mb = p.stat().st_size / 1e6
        ok(f"{fname} ({size_mb:.1f} MB)")
    else:
        warn(f"{fname} — NOT FOUND (train with: python scripts/train_all.py)")

# 6. External dependencies
print("\n[6] External tools")
wonder3d = Path("Wonder3D")
if wonder3d.exists() and (wonder3d / "test_mvdiffusion_seq.py").exists():
    ok("Wonder3D directory found")
else:
    warn("Wonder3D — NOT FOUND (needed for stages 6-7)")

# 7. Key Python files
print("\n[7] Pipeline scripts")
scripts = ["app.py", "generate_sd_only.py", "pixel2mesh_reconstruction.py",
           "render_mesh.py", "scripts/train_all.py", "scripts/run_full_pipeline.py"]
for s in scripts:
    if Path(s).exists():
        ok(s)
    else:
        fail(f"{s} — MISSING")
        errors += 1

# Summary
print("\n" + "=" * 50)
if errors == 0:
    print(f"  {GREEN}All checks passed!{RESET}")
    print(f"  Next steps:")
    print(f"    1. Place data files in ./data/")
    print(f"    2. Train models: python scripts/train_all.py")
    print(f"    3. Run pipeline: python scripts/run_full_pipeline.py")
else:
    print(f"  {RED}{errors} error(s) found{RESET} — fix before proceeding")
print("=" * 50)
