#!/usr/bin/env python
"""Train the unified mood classification model with a small amount of data"""

import argparse
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.models.unified_model import UnifiedMoodModel
from src.losses.multitask_loss import MultitaskLoss
from dataloader import get_dataloader

def main():
    # 1. 参数解析
    parser = argparse.ArgumentParser(description="Train the unified mood classification model with small data")
    parser.add_argument('--fusion_type', type=str, default='early', choices=['early', 'intermediate', 'late'], help='Feature fusion type')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension size')
    parser.add_argument('--num_transformer_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs (small number for testing)')
    parser.add_argument('--deam_weight', type=float, default=0.5, help='Weight for DEAM regression task')
    parser.add_argument('--mtg_weight', type=float, default=0.5, help='Weight for MTG-Jamendo classification task')
    parser.add_argument('--device', type=str, default='auto', help='Device to use for training (auto, mps, cpu, cuda)')
    parser.add_argument('--data_limit', type=int, default=4, help='Number of samples to use for each dataset')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')

    args = parser.parse_args()

    # 2. 设备配置 - 直接使用MPS
    device = torch.device('mps')
    print("Using MPS (Apple Silicon GPU)")

    # 3. 数据加载
    MERT_MODEL_PATH = "/Users/david/codespace/Unified-Mood-Classification-Mamba/MERT"

    # Load DEAM datasets (train/val/test) but with limited samples
    print("Loading DEAM dataset (limited samples)...")
    deam_train_loader = get_dataloader("deam", split="train", batch_size=args.batch_size, model_dir=MERT_MODEL_PATH)
    deam_val_loader = get_dataloader("deam", split="val", batch_size=args.batch_size, model_dir=MERT_MODEL_PATH)

    # Load MTG-Jamendo datasets (train/val/test) but with limited samples
    print("Loading MTG-Jamendo dataset (limited samples)...")
    mtg_train_loader = get_dataloader(
        "mtg-jamendo",
        split="train",
        batch_size=args.batch_size,
        model_dir=MERT_MODEL_PATH,
        num_workers=0
    )
    mtg_val_loader = get_dataloader(
        "mtg-jamendo",
        split="val",
        batch_size=args.batch_size,
        model_dir=MERT_MODEL_PATH,
        num_workers=0
    )

    # 4. 模型初始化
    print("Initializing model...")
    num_mtg_tags = len(mtg_train_loader.dataset.mood_tags)
    print(f"Number of MTG-Jamendo tags: {num_mtg_tags}")

    model = UnifiedMoodModel(
        fusion_type=args.fusion_type,
        hidden_dim=args.hidden_dim,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        num_class_tags=num_mtg_tags
    )
    model = model.to(device)  # Use full precision, let AMP handle automatic mixed precision

    # 5. 损失函数与优化器
    loss_fn = MultitaskLoss().to(device)  # Move loss to device
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # Initialize GradScaler for loss scaling with MPS support
    scaler = torch.amp.GradScaler(device=torch.device("cpu"))  # Fallback to cpu if MPS doesn't work

    # 6. TensorBoard 初始化
    log_dir = f"runs/unified_mood_model_small_{args.fusion_type}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # 7. 训练循环 - 使用少量数据
    print(f"Starting training with {args.epochs} epochs and limited data...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        deam_total = 0
        mtg_total = 0
        deam_ccc = 0.0

        # 交替训练两个数据集，但只使用有限样本
        deam_iter = iter(deam_train_loader)
        mtg_iter = iter(mtg_train_loader)
        data_count = 0  # Count of samples processed

        # Initialize gradient accumulation step counter
        accumulation_counter = 0

        while data_count < args.data_limit:
            try:
                # 训练DEAM batch
                deam_features, deam_labels = next(deam_iter)

                # Move data to device
                for key in deam_features:
                    deam_features[key] = deam_features[key].to(device)
                deam_labels = deam_labels.to(device)

                # Forward pass with automatic mixed precision
                with torch.amp.autocast(device_type='mps' if torch.backends.mps.is_available() else 'cpu', dtype=torch.float16):
                    outputs = model(deam_features)

                # Cast to float32 for loss calculation (if needed)
                outputs = {k: v.float() for k, v in outputs.items()}

                # Compute loss and metric
                loss, ccc = loss_fn(outputs, deam_labels, task_type='deam')

                # Normalize loss for gradient accumulation
                loss = loss / args.accumulation_steps

                # Backward pass with loss scaling
                scaler.scale(loss).backward()

                # Update statistics
                batch_size = deam_features['mel'].size(0)
                running_loss += (loss.item() * args.accumulation_steps) * batch_size
                deam_total += batch_size
                if ccc is not None:
                    deam_ccc += ccc.item() * batch_size

                data_count += batch_size
                accumulation_counter += 1

            except StopIteration:
                # Reinitialize DEAM iterator if exhausted
                deam_iter = iter(deam_train_loader)
                continue

            try:
                # 训练MTG batch
                mtg_features, mtg_labels = next(mtg_iter)

                # Move data to device
                for key in mtg_features:
                    mtg_features[key] = mtg_features[key].to(device)
                mtg_labels = mtg_labels.to(device)

                # Forward pass with automatic mixed precision
                with torch.amp.autocast(device_type='mps' if torch.backends.mps.is_available() else 'cpu', dtype=torch.float16):
                    outputs = model(mtg_features)

                # Cast to float32 for loss calculation (if needed)
                outputs = {k: v.float() for k, v in outputs.items()}

                # Compute loss
                loss, _ = loss_fn(outputs, mtg_labels, task_type='mtg')

                # Normalize loss for gradient accumulation
                loss = loss / args.accumulation_steps

                # Backward pass with loss scaling
                scaler.scale(loss).backward()

                # Update statistics
                batch_size = mtg_features['mel'].size(0)
                running_loss += (loss.item() * args.accumulation_steps) * batch_size
                mtg_total += batch_size

                data_count += batch_size
                accumulation_counter += 1

            except StopIteration:
                # Reinitialize MTG iterator if exhausted
                mtg_iter = iter(mtg_train_loader)
                continue

            # Perform optimizer step after accumulation_steps batches
            if accumulation_counter >= args.accumulation_steps:
                # Unscale gradients
                scaler.unscale_(optimizer)
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # Perform optimizer step with scaler
                scaler.step(optimizer)
                # Update scaler
                scaler.update()
                optimizer.zero_grad()
                accumulation_counter = 0

            if data_count >= args.data_limit:
                break

        # Make sure to take the last step if accumulation counter is not zero
        if accumulation_counter > 0:
            # Unscale gradients
            scaler.unscale_(optimizer)
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Perform optimizer step with scaler
            scaler.step(optimizer)
            # Update scaler
            scaler.update()
            optimizer.zero_grad()

        # Calculate epoch statistics
        epoch_loss = running_loss / (deam_total + mtg_total) if (deam_total + mtg_total) > 0 else 0.0
        avg_deam_ccc = deam_ccc / deam_total if deam_total > 0 else 0.0

        # Log to TensorBoard
        writer.add_scalar('Training/Loss', epoch_loss, epoch)
        writer.add_scalar('Training/DEAM_CCC', avg_deam_ccc, epoch)

        # Print training summary
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        print(f"Training Loss: {epoch_loss:.4f}")
        print(f"DEAM CCC: {avg_deam_ccc:.4f}")
        print(f"Samples processed: {data_count}")

    # 8. 模型保存与清理
    model_save_path = f"unified_mood_model_small_{args.fusion_type}.pt"
    torch.save(model.state_dict(), model_save_path)
    writer.close()

    print(f"\nTraining completed with limited data!")
    print(f"Model saved to: {model_save_path}")
    print(f"TensorBoard logs: {log_dir}")
    print("\nTo view TensorBoard:")
    print(f"tensorboard --logdir={log_dir}")

    # 测试模型是否能正常加载
    print("\nTesting model loading...")
    model_loaded = UnifiedMoodModel(
        fusion_type=args.fusion_type,
        num_class_tags=num_mtg_tags
    )
    model_loaded.load_state_dict(torch.load(model_save_path, map_location='cpu'))
    print("✅ Model loaded successfully!")

if __name__ == "__main__":
    main()
