#!/usr/bin/env python
"""Train the unified mood classification model with multitask learning"""

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
    parser = argparse.ArgumentParser(description="Train the unified mood classification model")
    parser.add_argument('--fusion_type', type=str, default='early', choices=['early', 'intermediate', 'late'], help='Feature fusion type')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension size')
    parser.add_argument('--num_transformer_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--deam_weight', type=float, default=0.5, help='Weight for DEAM regression task')
    parser.add_argument('--mtg_weight', type=float, default=0.5, help='Weight for MTG-Jamendo classification task')
    parser.add_argument('--device', type=str, default='auto', help='Device to use for training (auto, mps, cpu, cuda)')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')

    args = parser.parse_args()

    # 2. 设备配置 - 直接使用MPS
    device = torch.device('mps')
    print("Using MPS (Apple Silicon GPU)")

    # 3. 数据加载
    MERT_MODEL_PATH = "/Users/david/codespace/Unified-Mood-Classification-Mamba/MERT"

    # Load DEAM datasets (train/val/test)
    print("Loading DEAM dataset...")
    deam_train_loader = get_dataloader("deam", split="train", batch_size=args.batch_size, model_dir=MERT_MODEL_PATH)
    deam_val_loader = get_dataloader("deam", split="val", batch_size=args.batch_size, model_dir=MERT_MODEL_PATH)
    deam_test_loader = get_dataloader("deam", split="test", batch_size=args.batch_size, model_dir=MERT_MODEL_PATH)

    # Load MTG-Jamendo datasets (train/val/test)
    print("Loading MTG-Jamendo dataset...")
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
    mtg_test_loader = get_dataloader(
        "mtg-jamendo",
        split="test",
        batch_size=args.batch_size,
        model_dir=MERT_MODEL_PATH,
        num_workers=0
    )

    # 4. 模型初始化
    print("Initializing model...")
    model = UnifiedMoodModel(
        fusion_type=args.fusion_type,
        hidden_dim=args.hidden_dim,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads
    )
    model = model.to(device)  # Use full precision, let AMP handle automatic mixed precision

    # 5. 损失函数与优化器
    loss_fn = MultitaskLoss().to(device)  # Move loss to device
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # Initialize GradScaler for loss scaling (to handle float16 precision issues)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else torch.cpu.amp.GradScaler()

    # 6. TensorBoard 初始化
    log_dir = f"runs/unified_mood_model_{args.fusion_type}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # 7. 训练循环
    print(f"Starting training with {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        deam_total = 0
        mtg_total = 0
        deam_ccc = 0.0

        # 交替训练两个数据集
        deam_iter = iter(deam_train_loader)
        mtg_iter = iter(mtg_train_loader)
        num_batches = max(len(deam_train_loader), len(mtg_train_loader))

        # Initialize gradient accumulation step counter
        accumulation_counter = 0

        for batch_idx in range(num_batches):
            # 训练DEAM batch
            try:
                deam_features, deam_labels = next(deam_iter)

                # Move data to device
                for key in deam_features:
                    deam_features[key] = deam_features[key].to(device)
                deam_labels = deam_labels.to(device)

                # Forward pass with automatic mixed precision
                with torch.amp.autocast(device_type='mps' if torch.backends.mps.is_available() else 'cpu', dtype=torch.float16):
                    outputs = model(deam_features)

                # Compute loss and metric
                loss, ccc = loss_fn(outputs, deam_labels, task_type='deam')

                # Normalize loss for gradient accumulation
                loss = loss / args.accumulation_steps

                # Backward pass with loss scaling
                scaler.scale(loss).backward()

                # Update statistics
                running_loss += (loss.item() * args.accumulation_steps) * deam_features['mel'].size(0)
                deam_total += deam_features['mel'].size(0)
                if ccc is not None:
                    deam_ccc += ccc.item() * deam_features['mel'].size(0)

                # Update accumulation counter
                accumulation_counter += 1

            except StopIteration:
                # Reinitialize DEAM iterator if exhausted
                deam_iter = iter(deam_train_loader)
                continue

            # 训练MTG batch
            try:
                mtg_features, mtg_labels = next(mtg_iter)

                # Move data to device
                for key in mtg_features:
                    mtg_features[key] = mtg_features[key].to(device)
                mtg_labels = mtg_labels.to(device)

                # Forward pass - using half precision as model is already in half()
                outputs = model(mtg_features)

                # Compute loss
                loss, _ = loss_fn(outputs, mtg_labels, task_type='mtg')

                # Normalize loss for gradient accumulation
                loss = loss / args.accumulation_steps

                # Backward pass with loss scaling
                scaler.scale(loss).backward()

                # Update statistics
                running_loss += (loss.item() * args.accumulation_steps) * mtg_features['mel'].size(0)
                mtg_total += mtg_features['mel'].size(0)

                # Update accumulation counter
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

        # 8. 验证循环
        model.eval()
        val_loss = 0.0
        val_deam_total = 0
        val_mtg_total = 0
        val_deam_ccc = 0.0

        with torch.no_grad():
            # Validate DEAM
            for features, labels in deam_val_loader:
                for key in features:
                    features[key] = features[key].to(device)
                labels = labels.to(device)

                # Forward pass with automatic mixed precision
                with torch.amp.autocast(device_type='mps' if torch.backends.mps.is_available() else 'cpu', dtype=torch.float16):
                    outputs = model(features)
                loss, ccc = loss_fn(outputs, labels, task_type='deam')

                val_loss += loss.item() * features['mel'].size(0)
                val_deam_total += features['mel'].size(0)
                if ccc is not None:
                    val_deam_ccc += ccc.item() * features['mel'].size(0)

            # Validate MTG
            for features, labels in mtg_val_loader:
                for key in features:
                    features[key] = features[key].to(device)
                labels = labels.to(device)

                # Forward pass with automatic mixed precision
                with torch.amp.autocast(device_type='mps' if torch.backends.mps.is_available() else 'cpu', dtype=torch.float16):
                    outputs = model(features)
                loss, _ = loss_fn(outputs, labels, task_type='mtg')

                val_loss += loss.item() * features['mel'].size(0)
                val_mtg_total += features['mel'].size(0)

        # Calculate validation statistics
        val_epoch_loss = val_loss / (val_deam_total + val_mtg_total) if (val_deam_total + val_mtg_total) > 0 else 0.0
        val_avg_deam_ccc = val_deam_ccc / val_deam_total if val_deam_total > 0 else 0.0

        # Log validation to TensorBoard
        writer.add_scalar('Validation/Loss', val_epoch_loss, epoch)
        writer.add_scalar('Validation/DEAM_CCC', val_avg_deam_ccc, epoch)

        # Print validation summary
        print(f"Validation Loss: {val_epoch_loss:.4f}")
        print(f"Validation DEAM CCC: {val_avg_deam_ccc:.4f}")
        print("-" * 60)

    # 9. 测试阶段
    print("\nStarting final testing...")
    model.eval()
    test_loss = 0.0
    test_deam_total = 0
    test_mtg_total = 0
    test_deam_ccc = 0.0

    with torch.no_grad():
        # Test DEAM
        for features, labels in deam_test_loader:
            for key in features:
                features[key] = features[key].to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss, ccc = loss_fn(outputs, labels, task_type='deam')

            test_loss += loss.item() * features['mel'].size(0)
            test_deam_total += features['mel'].size(0)
            if ccc is not None:
                test_deam_ccc += ccc.item() * features['mel'].size(0)

        # Test MTG
        for features, labels in mtg_test_loader:
            for key in features:
                features[key] = features[key].to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss, _ = loss_fn(outputs, labels, task_type='mtg')

            test_loss += loss.item() * features['mel'].size(0)
            test_mtg_total += features['mel'].size(0)

    # Calculate test statistics
    test_epoch_loss = test_loss / (test_deam_total + test_mtg_total) if (test_deam_total + test_mtg_total) > 0 else 0.0
    test_avg_deam_ccc = test_deam_ccc / test_deam_total if test_deam_total > 0 else 0.0

    # Log test to TensorBoard
    writer.add_scalar('Test/Loss', test_epoch_loss, args.epochs)
    writer.add_scalar('Test/DEAM_CCC', test_avg_deam_ccc, args.epochs)

    # Print test summary
    print(f"Test Loss: {test_epoch_loss:.4f}")
    print(f"Test DEAM CCC: {test_avg_deam_ccc:.4f}")
    print("-" * 60)

    # 10. 模型保存与清理
    model_save_path = f"unified_mood_model_{args.fusion_type}.pt"
    torch.save(model.state_dict(), model_save_path)
    writer.close()

    print(f"\nTraining completed!")
    print(f"Model saved to: {model_save_path}")
    print(f"TensorBoard logs: {log_dir}")
    print("\nTo view TensorBoard:")
    print(f"tensorboard --logdir={log_dir}")

if __name__ == "__main__":
    main()
