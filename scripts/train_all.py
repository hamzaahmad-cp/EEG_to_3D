#!/usr/bin/env python3
"""
Train both EEG models (classifier + contrastive encoder) sequentially.

Usage:
    python scripts/train_all.py
    python scripts/train_all.py --config config/config.yaml
    python scripts/train_all.py --skip-classifier   # only train contrastive
    python scripts/train_all.py --skip-contrastive   # only train classifier
"""

import sys
import os
import argparse

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description="Train EEG models for EEG-to-3D pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--skip-classifier", action="store_true")
    parser.add_argument("--skip-contrastive", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Step 1: Train EEG Classifier ─────────────────────────────
    if not args.skip_classifier:
        print("\n" + "=" * 60)
        print("  STAGE 1: Training EEG Classifier")
        print("=" * 60)

        from src.models.eeg_classifier import EEGClassifier
        from src.data.data_loader import CATVisDataLoader
        from src.data.preprocessor import DataPreprocessor

        data_loader = CATVisDataLoader(config)
        df = data_loader.get_dataset_dataframe()
        train_df, val_df, test_df = data_loader.get_train_val_test_splits(df)
        print(f"Splits — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        preprocessor = DataPreprocessor(config)
        train_ds, val_ds, test_ds = preprocessor.create_classification_datasets(train_df, val_df, test_df)
        train_loader, val_loader, test_loader = preprocessor.create_data_loaders(train_ds, val_ds, test_ds)

        model = EEGClassifier(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['eeg_classification']['learning_rate'])
        criterion = torch.nn.CrossEntropyLoss()

        best_val_acc = 0.0
        ckpt_path = os.path.join(config['checkpoints']['root_dir'], config['checkpoints']['eeg_classifier'])
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        for epoch in range(config['eeg_classification']['num_epochs']):
            # Train
            model.train()
            correct = total = 0
            running_loss = 0.0
            for eeg, label in train_loader:
                eeg, label = eeg.to(device), label.to(device)
                optimizer.zero_grad()
                outputs, _ = model(eeg)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                correct += (outputs.argmax(dim=-1) == label).sum().item()
                total += label.size(0)
            train_acc = 100 * correct / total

            # Validate
            model.eval()
            val_correct = val_total = 0
            val_loss = 0.0
            with torch.no_grad():
                for eeg, label in val_loader:
                    eeg, label = eeg.to(device), label.to(device)
                    outputs, _ = model(eeg)
                    val_loss += criterion(outputs, label).item()
                    val_correct += (outputs.argmax(dim=-1) == label).sum().item()
                    val_total += label.size(0)
            val_acc = 100 * val_correct / val_total

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{config['eeg_classification']['num_epochs']}  "
                      f"TrL={running_loss/len(train_loader):.4f}  TrA={train_acc:.1f}%  "
                      f"VaL={val_loss/len(val_loader):.4f}  VaA={val_acc:.1f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.model.state_dict(), ckpt_path)

        print(f"  Best val accuracy: {best_val_acc:.2f}%  →  {ckpt_path}")

        # Test
        model.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for eeg, label in test_loader:
                eeg, label = eeg.to(device), label.to(device)
                outputs, _ = model(eeg)
                test_correct += (outputs.argmax(dim=-1) == label).sum().item()
                test_total += label.size(0)
        print(f"  Test accuracy: {100*test_correct/test_total:.2f}%")

    # ── Step 2: Train Contrastive Encoder ────────────────────────
    if not args.skip_contrastive:
        print("\n" + "=" * 60)
        print("  STAGE 2: Training Contrastive Encoder")
        print("=" * 60)

        import torch.nn.functional as F
        import clip
        from src.models.contrastive_encoder import ContrastiveEncoder, clip_style_contrastive_loss
        from src.data.data_loader import CATVisDataLoader
        from src.data.preprocessor import DataPreprocessor

        if args.skip_classifier:
            data_loader = CATVisDataLoader(config)
            df = data_loader.get_dataset_dataframe()
            train_df, val_df, test_df = data_loader.get_train_val_test_splits(df)

        preprocessor = DataPreprocessor(config)
        train_ds, val_ds, _ = preprocessor.create_contrastive_datasets(train_df, val_df, test_df)
        batch_size = config['contrastive_training']['batch_size']
        train_loader, val_loader, _ = preprocessor.create_data_loaders(
            train_ds, val_ds, val_ds, batch_size=batch_size
        )

        eeg_model = ContrastiveEncoder(config).to(device)

        # Warm-start from classifier backbone
        classifier_ckpt = os.path.join(config['checkpoints']['root_dir'], config['checkpoints']['eeg_classifier'])
        if os.path.exists(classifier_ckpt):
            clf_state = torch.load(classifier_ckpt, map_location=device)
            matched, total_keys = 0, len(clf_state)
            model_state = eeg_model.model.state_dict()
            for k, v in clf_state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    model_state[k] = v
                    matched += 1
            eeg_model.model.load_state_dict(model_state)
            print(f"  Warm-started from classifier: {matched}/{total_keys} keys matched")

        clip_model, _ = clip.load(config['contrastive_training']['clip_model'], device=device)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        optimizer = torch.optim.Adam(eeg_model.parameters(), lr=config['contrastive_training']['learning_rate'])
        temperature = config['contrastive_training']['temperature']
        patience = config['contrastive_training']['patience']
        best_val_loss = float('inf')
        no_improve = 0

        ckpt_path = os.path.join(config['checkpoints']['root_dir'], config['checkpoints']['contrastive_model'])

        for epoch in range(config['contrastive_training']['num_epochs']):
            # Train
            eeg_model.train()
            train_losses = []
            for eeg_batch, text_batch in train_loader:
                eeg_batch = eeg_batch.to(device)
                with torch.no_grad():
                    text_tokens = clip.tokenize(text_batch, truncate=True).to(device)
                    text_embeds = clip_model.encode_text(text_tokens).float()
                    text_embeds = F.normalize(text_embeds, dim=-1)
                eeg_embeds = eeg_model(eeg_batch)
                eeg_embeds = F.normalize(eeg_embeds, dim=-1)
                loss, _ = clip_style_contrastive_loss(eeg_embeds, text_embeds, temperature)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # Validate
            eeg_model.eval()
            val_losses = []
            with torch.no_grad():
                for eeg_batch, text_batch in val_loader:
                    eeg_batch = eeg_batch.to(device)
                    text_tokens = clip.tokenize(text_batch, truncate=True).to(device)
                    text_embeds = clip_model.encode_text(text_tokens).float()
                    text_embeds = F.normalize(text_embeds, dim=-1)
                    eeg_embeds = eeg_model(eeg_batch)
                    eeg_embeds = F.normalize(eeg_embeds, dim=-1)
                    loss, _ = clip_style_contrastive_loss(eeg_embeds, text_embeds, temperature)
                    val_losses.append(loss.item())

            avg_train = sum(train_losses) / len(train_losses)
            avg_val = sum(val_losses) / len(val_losses)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{config['contrastive_training']['num_epochs']}  "
                      f"TrL={avg_train:.4f}  VaL={avg_val:.4f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                no_improve = 0
                torch.save(eeg_model.model.state_dict(), ckpt_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        print(f"  Best val loss: {best_val_loss:.4f}  →  {ckpt_path}")

    print("\nDone. Both models trained.")


if __name__ == "__main__":
    main()
