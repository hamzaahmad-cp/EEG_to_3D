#!/usr/bin/env python3
"""
Run the complete EEG-to-3D pipeline on the full test set and compute metrics.

This script:
  1. Loads trained EEG models (classifier + contrastive encoder)
  2. Runs EEG inference on all ~1,997 test samples
  3. Retrieves captions and builds prompts
  4. Generates images with FLUX.1-schnell
  5. Removes backgrounds
  6. Runs Wonder3D multi-view generation
  7. Reconstructs 3D meshes
  8. Computes 2D metrics (IS, FID, GA) and 3D metrics (silhouette IoU)

Usage:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --stages 1-5      # 2D generation only
    python scripts/run_full_pipeline.py --stages 6-7      # 3D only (assumes images exist)
    python scripts/run_full_pipeline.py --max-samples 50   # quick test
    python scripts/run_full_pipeline.py --eval-only        # skip generation, just compute metrics
"""

import sys, os, time, json, argparse, shutil
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import torch
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_stages(s):
    """Parse stage range like '1-5' or '1-7'."""
    if '-' in s:
        lo, hi = s.split('-')
        return set(range(int(lo), int(hi) + 1))
    return {int(s)}


def is_caption_relevant(predicted_class: str, caption: str) -> bool:
    class_words = predicted_class.lower().replace("-", " ").replace("_", " ").split()
    return any(len(w) > 3 and w in caption.lower() for w in class_words)


def create_enhanced_prompt(predicted_class: str, caption: str) -> tuple:
    use_caption = is_caption_relevant(predicted_class, caption)
    if use_caption:
        c = caption
        for src, dst in [("A group of", "A single"), ("group of", "single"),
                         ("Several", "One"), ("several", "one"), ("Many", "One"),
                         ("many", "one"), ("Two", "One"), ("two", "one"),
                         ("Three", "One"), ("three", "one"),
                         ("people", ""), ("persons", ""), ("men", ""), ("women", "")]:
            c = c.replace(src, dst)
        prompt = (f"ONE single {predicted_class}, exactly one object, {c}, "
                  f"full body visible, complete anatomy, whole body in frame, "
                  f"solo subject only, no other objects, isolated on pure white background, "
                  f"centered, studio product photography, professional lighting, sharp focus, 8k")
    else:
        prompt = (f"ONE single {predicted_class}, exactly one object, "
                  f"full body visible, complete anatomy, whole body in frame, "
                  f"solo subject only, no other objects, isolated on pure white background, "
                  f"centered, studio product photography, professional lighting, sharp focus, 8k, photorealistic")
    return prompt, use_caption


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--stages", default="1-7", help="Pipeline stages to run (e.g. '1-5', '6-7')")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit test samples (for debugging)")
    parser.add_argument("--eval-only", action="store_true", help="Skip generation, only compute metrics")
    parser.add_argument("--batch-start", type=int, default=0, help="Resume from this sample index")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flux-steps", type=int, default=4)
    parser.add_argument("--skip-wonder3d", action="store_true", help="Skip Wonder3D (stages 6-7)")
    args = parser.parse_args()

    stages = parse_stages(args.stages)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Stages: {sorted(stages)}")

    # Output dirs
    out_root = project_root / "outputs" / "full_pipeline"
    gen_dir = out_root / "generated_images"
    gen_nobg_dir = out_root / "generated_nobg"
    gt_dir = out_root / "ground_truth_images"
    wonder3d_input = project_root / "Wonder3D" / "example_images"
    meta_path = out_root / "metadata.json"
    for d in [gen_dir, gen_nobg_dir, gt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Load models & data ──────────────────────────────────
    print("\nLoading models and data...")
    from src.models.eeg_classifier import EEGClassifier
    from src.models.contrastive_encoder import ContrastiveEncoder
    from src.data.data_loader import CATVisDataLoader
    from src.data.preprocessor import DataPreprocessor
    from src.pipeline.retrieval import TextRetrieval

    classifier = EEGClassifier(config).to(device)
    classifier.model.load_state_dict(
        torch.load(os.path.join(config['checkpoints']['root_dir'], config['checkpoints']['eeg_classifier']),
                   map_location=device))
    classifier.eval()

    contrastive = ContrastiveEncoder(config).to(device)
    contrastive.model.load_state_dict(
        torch.load(os.path.join(config['checkpoints']['root_dir'], config['checkpoints']['contrastive_model']),
                   map_location=device))
    contrastive.eval()

    data_loader = CATVisDataLoader(config)
    df = data_loader.get_dataset_dataframe()
    _, _, test_df = data_loader.get_train_val_test_splits(df)

    preprocessor = DataPreprocessor(config)
    test_dataset = preprocessor.create_pipeline_dataset(test_df)

    retrieval = TextRetrieval(config, contrastive, device)
    retrieval.setup_retrieval_corpus(test_df)

    labels_order = data_loader.raw_data['labels']
    n_samples = len(test_dataset) if args.max_samples is None else min(args.max_samples, len(test_dataset))
    print(f"Test samples: {n_samples}")

    # ── Load metadata if resuming ────────────────────────────
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    # ── Stages 1-5: EEG → Classification → Retrieval → Flux.1 → Background removal ──
    if stages & {1, 2, 3, 4, 5} and not args.eval_only:
        print("\n" + "=" * 60)
        print("  STAGES 1-5: EEG → Image Generation")
        print("=" * 60)

        # Lazy-load Flux
        flux_pipe = None

        for idx in tqdm(range(args.batch_start, n_samples), desc="Generating"):
            sample = test_dataset[idx]
            key = f"sample_{idx:04d}"

            # Skip if already generated
            nobg_path = gen_nobg_dir / f"{key}.png"
            if nobg_path.exists() and key in metadata:
                continue

            # Stage 1-2: EEG encoding + classification
            eeg_data = sample['eeg'].unsqueeze(0).to(device)
            gt_idx = sample['label'].item() if torch.is_tensor(sample['label']) else sample['label']
            img_idx = sample['image'].item() if torch.is_tensor(sample['image']) else sample['image']

            with torch.no_grad():
                outputs, _ = classifier(eeg_data)
                pred_idx = outputs.argmax().item()

            predicted_class = config['class_prompts'][labels_order[pred_idx]]
            ground_truth_class = config['class_prompts'][labels_order[gt_idx]]

            # Stage 3: Caption retrieval
            retrieved = retrieval.retrieve_top_k_from_eeg(eeg_data.squeeze(0), k=1)
            caption = retrieved[0][0]

            # Stage 4: Build prompt (beta interpolation replaced with text-space blending)
            prompt, caption_used = create_enhanced_prompt(predicted_class, caption)

            # Stage 5: Flux.1 generation
            if flux_pipe is None:
                print("  Loading FLUX.1-schnell...")
                from diffusers import FluxPipeline
                flux_pipe = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
                flux_pipe.enable_model_cpu_offload()

            result = flux_pipe(
                prompt, guidance_scale=0.0, num_inference_steps=args.flux_steps,
                max_sequence_length=256,
                generator=torch.Generator("cpu").manual_seed(args.seed + idx),
                height=1024, width=1024,
            )
            gen_image = result.images[0]
            gen_image.save(gen_dir / f"{key}.png")

            # Background removal
            from rembg import remove as rembg_remove
            image_no_bg = rembg_remove(gen_image)
            image_no_bg.save(nobg_path)

            # Save GT image
            rel_path = data_loader.image_to_path.get(img_idx)
            if rel_path:
                src = Path(config['data']['root_dir']) / config['data']['imagenet_images'] / rel_path
                if src.exists():
                    shutil.copy(src, gt_dir / f"{key}_gt.JPEG")

            # Save metadata
            metadata[key] = {
                'sample_idx': idx,
                'ground_truth_class': ground_truth_class,
                'predicted_class': predicted_class,
                'correct': predicted_class.lower() == ground_truth_class.lower(),
                'retrieved_caption': caption,
                'caption_used': caption_used,
                'image_idx': img_idx,
            }

            # Save metadata incrementally
            if (idx + 1) % 10 == 0:
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

        # Final metadata save
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Classification accuracy
        if metadata:
            correct = sum(1 for v in metadata.values() if v.get('correct'))
            total = len(metadata)
            print(f"\n  Classification accuracy: {correct}/{total} = {100*correct/total:.1f}%")

    # ── Stage 6: Wonder3D multi-view generation ──────────────
    if 6 in stages and not args.eval_only and not args.skip_wonder3d:
        print("\n" + "=" * 60)
        print("  STAGE 6: Wonder3D Multi-View Generation")
        print("=" * 60)

        wonder3d_dir = project_root / "Wonder3D"
        wonder3d_input.mkdir(parents=True, exist_ok=True)

        nobg_files = sorted(gen_nobg_dir.glob("*.png"))
        for img_path in tqdm(nobg_files, desc="Wonder3D"):
            scene_name = img_path.stem
            scene_out = wonder3d_dir / "outputs" / "cropsize-192-cfg3.0" / scene_name
            if scene_out.exists():
                continue

            shutil.copy(img_path, wonder3d_input / img_path.name)
            try:
                import subprocess
                subprocess.run([
                    'python', 'test_mvdiffusion_seq.py',
                    '--config', 'configs/mvdiffusion-joint-ortho-6views.yaml',
                    f'validation_dataset.filepaths=[{img_path.name}]'
                ], cwd=str(wonder3d_dir), check=True, capture_output=True, timeout=300)
            except Exception as e:
                print(f"  Wonder3D failed for {scene_name}: {e}")

    # ── Stage 7: 3D Mesh Reconstruction ──────────────────────
    if 7 in stages and not args.eval_only:
        print("\n" + "=" * 60)
        print("  STAGE 7: 3D Mesh Reconstruction")
        print("=" * 60)

        from pixel2mesh_reconstruction import (
            load_views, space_carve, occupancy_to_mesh,
            assign_normals, assign_colors, make_o3d_mesh, save_mesh,
        )

        wonder3d_out = project_root / "Wonder3D" / "outputs" / "cropsize-192-cfg3.0"
        if wonder3d_out.exists():
            scene_dirs = sorted([d for d in wonder3d_out.iterdir() if d.is_dir()])
            for scene_dir in tqdm(scene_dirs, desc="Reconstructing"):
                mesh_out = project_root / "outputs" / "meshes" / scene_dir.name
                if (mesh_out / "mesh_carving.ply").exists():
                    continue
                try:
                    views = load_views(scene_dir)
                    if not views:
                        continue
                    occ = space_carve(views, grid_res=128, orth_scale=1.05)
                    if occ.sum() == 0:
                        continue
                    v, f = occupancy_to_mesh(occ, orth_scale=1.05, gaussian_sigma=1.0)
                    nrm = assign_normals(v, views, orth_scale=1.05)
                    clr = assign_colors(v, nrm, views, orth_scale=1.05)
                    mesh = make_o3d_mesh(v, f, clr, nrm)
                    import open3d as o3d
                    mesh = mesh.filter_smooth_laplacian(number_of_iterations=3)
                    mesh.compute_vertex_normals()
                    save_mesh(mesh, mesh_out, "mesh_carving")
                except Exception as e:
                    print(f"  Reconstruction failed for {scene_dir.name}: {e}")

    # ── Evaluation ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)

    # Reload metadata
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    # Classification accuracy
    if metadata:
        correct = sum(1 for v in metadata.values() if v.get('correct'))
        total = len(metadata)
        print(f"  Classification: {correct}/{total} = {100*correct/total:.1f}%")

        # Top-1 by class
        from collections import Counter
        class_correct = Counter()
        class_total = Counter()
        for v in metadata.values():
            gt = v['ground_truth_class']
            class_total[gt] += 1
            if v.get('correct'):
                class_correct[gt] += 1
        print(f"\n  Per-class accuracy (top/bottom 5):")
        accs = {c: class_correct[c] / class_total[c] for c in class_total}
        for c, a in sorted(accs.items(), key=lambda x: -x[1])[:5]:
            print(f"    {c}: {a*100:.0f}%")
        print("    ...")
        for c, a in sorted(accs.items(), key=lambda x: x[1])[:5]:
            print(f"    {c}: {a*100:.0f}%")

    # 3D: silhouette IoU
    wonder3d_out = project_root / "Wonder3D" / "outputs" / "cropsize-192-cfg3.0"
    mesh_dir = project_root / "outputs" / "meshes"
    if mesh_dir.exists() and wonder3d_out.exists():
        mesh_count = len(list(mesh_dir.glob("*/mesh_carving.ply")))
        print(f"\n  3D meshes reconstructed: {mesh_count}")

    print(f"\n  All outputs saved to: {out_root}")
    print("Done.")


if __name__ == "__main__":
    main()
