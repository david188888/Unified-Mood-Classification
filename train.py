#!/usr/bin/env python
"""Train the unified mood classification model with multitask learning"""

import argparse
from contextlib import nullcontext
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from src.models.unified_model import UnifiedMoodModel
from src.losses.multitask_loss import MultitaskLoss
from dataloader import get_dataloader


def _ccc_np(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    # pred/target: [N, D]
    pred_mean = pred.mean(axis=0)
    target_mean = target.mean(axis=0)
    pred_var = ((pred - pred_mean) ** 2).mean(axis=0)
    target_var = ((target - target_mean) ** 2).mean(axis=0)
    cov = ((pred - pred_mean) * (target - target_mean)).mean(axis=0)
    ccc = (2.0 * cov) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + eps)
    return float(np.nanmean(ccc))


def _pearson_np(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    # pred/target: [N, D]
    pred_mean = pred.mean(axis=0)
    target_mean = target.mean(axis=0)
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    cov = (pred_centered * target_centered).mean(axis=0)
    pred_std = np.sqrt((pred_centered**2).mean(axis=0))
    target_std = np.sqrt((target_centered**2).mean(axis=0))
    r = cov / (pred_std * target_std + eps)
    return float(np.nanmean(r))


def _rmse_np(pred: np.ndarray, target: np.ndarray) -> float:
    # pred/target: [N, D]
    rmse_per_dim = np.sqrt(((pred - target) ** 2).mean(axis=0))
    return float(np.nanmean(rmse_per_dim))


def _mtg_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute multilabel metrics.

    y_true: [N, C] in {0,1}
    y_score: [N, C] in [0,1]
    """
    metrics: dict[str, float] = {}
    y_pred = (y_score >= threshold).astype(np.int32)

    # F1
    metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Precision / Recall
    metrics["precision_micro"] = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["recall_micro"] = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # ROC-AUC
    try:
        metrics["roc_auc_micro"] = float(roc_auc_score(y_true, y_score, average="micro"))
    except ValueError:
        metrics["roc_auc_micro"] = float("nan")

    per_class_roc = []
    for c in range(y_true.shape[1]):
        # ROC-AUC undefined if the class is all-0 or all-1
        if np.unique(y_true[:, c]).size < 2:
            continue
        try:
            per_class_roc.append(float(roc_auc_score(y_true[:, c], y_score[:, c])))
        except ValueError:
            continue
    metrics["roc_auc_macro"] = float(np.mean(per_class_roc)) if per_class_roc else float("nan")

    # PR-AUC (Average Precision)
    try:
        metrics["pr_auc_micro"] = float(average_precision_score(y_true, y_score, average="micro"))
    except ValueError:
        metrics["pr_auc_micro"] = float("nan")

    try:
        metrics["pr_auc_macro"] = float(average_precision_score(y_true, y_score, average="macro"))
    except ValueError:
        # Fall back to manual macro that skips undefined classes (no positive samples)
        per_class_ap = []
        for c in range(y_true.shape[1]):
            if y_true[:, c].sum() == 0:
                continue
            try:
                per_class_ap.append(float(average_precision_score(y_true[:, c], y_score[:, c])))
            except ValueError:
                continue
        metrics["pr_auc_macro"] = float(np.mean(per_class_ap)) if per_class_ap else float("nan")

    return metrics

def main():
    # 1. 参数解析
    parser = argparse.ArgumentParser(description="Train the unified mood classification model")
    parser.add_argument('--fusion_type', type=str, default='early', choices=['early', 'late'], help='Feature fusion type')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension size')
    parser.add_argument('--num_transformer_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training') 
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--deam_weight', type=float, default=0.5, help='Weight for DEAM regression task')
    parser.add_argument('--mtg_weight', type=float, default=0.5, help='Weight for MTG-Jamendo classification task')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--train_pct', type=float, default=1.0, help='Use only this fraction of training data (0-1]')
    parser.add_argument('--subset_seed', type=int, default=42, help='Seed for dataset sub-sampling')

    args = parser.parse_args()

    # 2. 设备配置（mac 优先使用 MPS）
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    use_amp = device.type == "mps"
    autocast_ctx = torch.amp.autocast(device_type="mps", dtype=torch.float16) if use_amp else nullcontext()

    # 3. 数据加载
    MERT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MERT")

    # Load DEAM datasets (train/val/test)
    print("Loading DEAM dataset...")
    deam_train_loader = get_dataloader(
        "deam",
        split="train",
        batch_size=args.batch_size,
        model_dir=MERT_MODEL_PATH,
        subset_fraction=args.train_pct,
        subset_seed=args.subset_seed,
    )
    deam_val_loader = get_dataloader("deam", split="val", batch_size=args.batch_size, model_dir=MERT_MODEL_PATH, shuffle=False)
    deam_test_loader = get_dataloader("deam", split="test", batch_size=args.batch_size, model_dir=MERT_MODEL_PATH, shuffle=False)

    # Load MTG-Jamendo datasets (train/val/test)
    print("Loading MTG-Jamendo dataset...")
    mtg_train_loader = get_dataloader(
        "mtg-jamendo",
        split="train",
        batch_size=args.batch_size,
        model_dir=MERT_MODEL_PATH,
        num_workers=0,
        subset_fraction=args.train_pct,
        subset_seed=args.subset_seed,
    )
    mtg_val_loader = get_dataloader(
        "mtg-jamendo",
        split="val",
        batch_size=args.batch_size,
        model_dir=MERT_MODEL_PATH,
        num_workers=0,
        shuffle=False
    )
    mtg_test_loader = get_dataloader(
        "mtg-jamendo",
        split="test",
        batch_size=args.batch_size,
        model_dir=MERT_MODEL_PATH,
        num_workers=0,
        shuffle=False
    )

    num_mtg_tags = len(getattr(mtg_train_loader.dataset, "mood_tags", []))
    if num_mtg_tags <= 0:
        raise RuntimeError("MTG dataset has no mood tags; cannot determine classification head size")

    # 4. 模型初始化
    print("Initializing model...")
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
                with autocast_ctx:
                    outputs = model(deam_features)

                # Compute loss and metric
                loss, ccc = loss_fn(outputs, deam_labels, task_type='deam')

                # Task weight
                loss = loss * args.deam_weight

                # Normalize loss for gradient accumulation
                loss = loss / args.accumulation_steps

                loss.backward()

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

                # Forward pass with automatic mixed precision
                with autocast_ctx:
                    outputs = model(mtg_features)

                # Compute loss
                loss, _ = loss_fn(outputs, mtg_labels, task_type='mtg')

                # Task weight
                loss = loss * args.mtg_weight
                

                # Normalize loss for gradient accumulation
                loss = loss / args.accumulation_steps

                loss.backward()

                # Update statistics
                running_loss += (loss.item() * args.accumulation_steps) * mtg_features['mel'].size(0)
                mtg_total += mtg_features['mel'].size(0)

                print(f"\rEpoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{num_batches}]", end='')
                
                
                # Update accumulation counter
                accumulation_counter += 1

            except StopIteration:
                # Reinitialize MTG iterator if exhausted
                mtg_iter = iter(mtg_train_loader)
                continue

            # Perform optimizer step after accumulation_steps batches
            if accumulation_counter >= args.accumulation_steps:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                accumulation_counter = 0

        # Make sure to take the last step if accumulation counter is not zero
        if accumulation_counter > 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
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

        deam_val_preds = []
        deam_val_targets = []
        mtg_val_scores = []
        mtg_val_targets = []

        with torch.no_grad():
            # Validate DEAM
            for features, labels in deam_val_loader:
                for key in features:
                    features[key] = features[key].to(device)
                labels = labels.to(device)

                # Forward pass with automatic mixed precision
                with autocast_ctx:
                    outputs = model(features)
                loss, ccc = loss_fn(outputs, labels, task_type='deam')

                loss = loss * args.deam_weight

                val_loss += loss.item() * features['mel'].size(0)
                val_deam_total += features['mel'].size(0)
                if ccc is not None:
                    val_deam_ccc += ccc.item() * features['mel'].size(0)

                deam_val_preds.append(outputs['regression'].detach().cpu().float().numpy())
                deam_val_targets.append(labels.detach().cpu().float().numpy())

            # Validate MTG
            for features, labels in mtg_val_loader:
                for key in features:
                    features[key] = features[key].to(device)
                labels = labels.to(device)

                # Forward pass with automatic mixed precision
                with autocast_ctx:
                    outputs = model(features)
                loss, _ = loss_fn(outputs, labels, task_type='mtg')

                loss = loss * args.mtg_weight

                val_loss += loss.item() * features['mel'].size(0)
                val_mtg_total += features['mel'].size(0)

                mtg_logits = outputs['classification'].detach().cpu().float().numpy()
                mtg_probs = 1.0 / (1.0 + np.exp(-mtg_logits))
                mtg_val_scores.append(mtg_probs)
                mtg_val_targets.append(labels.detach().cpu().numpy())

        # Calculate validation statistics
        val_epoch_loss = val_loss / (val_deam_total + val_mtg_total) if (val_deam_total + val_mtg_total) > 0 else 0.0
        val_avg_deam_ccc = val_deam_ccc / val_deam_total if val_deam_total > 0 else 0.0

        # Log validation to TensorBoard
        writer.add_scalar('Validation/Loss', val_epoch_loss, epoch)
        writer.add_scalar('Validation/DEAM_CCC', val_avg_deam_ccc, epoch)

        # Compute epoch-level DEAM metrics on full validation set
        if deam_val_preds and deam_val_targets:
            deam_pred = np.concatenate(deam_val_preds, axis=0)
            deam_tgt = np.concatenate(deam_val_targets, axis=0)
            writer.add_scalar('Validation/DEAM_RMSE', _rmse_np(deam_pred, deam_tgt), epoch)
            writer.add_scalar('Validation/DEAM_Pearson', _pearson_np(deam_pred, deam_tgt), epoch)
            writer.add_scalar('Validation/DEAM_CCC_epoch', _ccc_np(deam_pred, deam_tgt), epoch)

        # Compute epoch-level MTG metrics on full validation set
        if mtg_val_scores and mtg_val_targets:
            y_score = np.concatenate(mtg_val_scores, axis=0)
            y_true = np.concatenate(mtg_val_targets, axis=0).astype(np.int32)
            mtg_m = _mtg_metrics(y_true=y_true, y_score=y_score, threshold=0.5)
            writer.add_scalar('Validation/MTG_ROC_AUC_micro', mtg_m['roc_auc_micro'], epoch)
            writer.add_scalar('Validation/MTG_ROC_AUC_macro', mtg_m['roc_auc_macro'], epoch)
            writer.add_scalar('Validation/MTG_PR_AUC_micro', mtg_m['pr_auc_micro'], epoch)
            writer.add_scalar('Validation/MTG_PR_AUC_macro', mtg_m['pr_auc_macro'], epoch)
            writer.add_scalar('Validation/MTG_F1_micro', mtg_m['f1_micro'], epoch)
            writer.add_scalar('Validation/MTG_F1_macro', mtg_m['f1_macro'], epoch)
            writer.add_scalar('Validation/MTG_Precision_micro', mtg_m['precision_micro'], epoch)
            writer.add_scalar('Validation/MTG_Precision_macro', mtg_m['precision_macro'], epoch)
            writer.add_scalar('Validation/MTG_Recall_micro', mtg_m['recall_micro'], epoch)
            writer.add_scalar('Validation/MTG_Recall_macro', mtg_m['recall_macro'], epoch)

        # Print validation summary
        print(f"Validation Loss: {val_epoch_loss:.4f}")
        print(f"Validation DEAM CCC: {val_avg_deam_ccc:.4f}")
        if deam_val_preds and deam_val_targets:
            print(f"Validation DEAM RMSE: {_rmse_np(deam_pred, deam_tgt):.4f}")
            print(f"Validation DEAM Pearson: {_pearson_np(deam_pred, deam_tgt):.4f}")
            print(f"Validation DEAM CCC (epoch): {_ccc_np(deam_pred, deam_tgt):.4f}")
        if mtg_val_scores and mtg_val_targets:
            print(
                "Validation MTG "
                f"ROC-AUC micro/macro: {mtg_m['roc_auc_micro']:.4f}/{mtg_m['roc_auc_macro']:.4f} | "
                f"PR-AUC micro/macro: {mtg_m['pr_auc_micro']:.4f}/{mtg_m['pr_auc_macro']:.4f} | "
                f"F1 micro/macro: {mtg_m['f1_micro']:.4f}/{mtg_m['f1_macro']:.4f} | "
                f"P micro/macro: {mtg_m['precision_micro']:.4f}/{mtg_m['precision_macro']:.4f} | "
                f"R micro/macro: {mtg_m['recall_micro']:.4f}/{mtg_m['recall_macro']:.4f}"
            )
        print("-" * 60)

    # 9. 测试阶段
    print("\nStarting final testing...")
    model.eval()
    test_loss = 0.0
    test_deam_total = 0
    test_mtg_total = 0
    test_deam_ccc = 0.0

    deam_test_preds = []
    deam_test_targets = []
    mtg_test_scores = []
    mtg_test_targets = []

    with torch.no_grad():
        # Test DEAM
        for features, labels in deam_test_loader:
            for key in features:
                features[key] = features[key].to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss, ccc = loss_fn(outputs, labels, task_type='deam')

            loss = loss * args.deam_weight

            test_loss += loss.item() * features['mel'].size(0)
            test_deam_total += features['mel'].size(0)
            if ccc is not None:
                test_deam_ccc += ccc.item() * features['mel'].size(0)

            deam_test_preds.append(outputs['regression'].detach().cpu().float().numpy())
            deam_test_targets.append(labels.detach().cpu().float().numpy())

        # Test MTG
        for features, labels in mtg_test_loader:
            for key in features:
                features[key] = features[key].to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss, _ = loss_fn(outputs, labels, task_type='mtg')

            loss = loss * args.mtg_weight

            test_loss += loss.item() * features['mel'].size(0)
            test_mtg_total += features['mel'].size(0)

            mtg_logits = outputs['classification'].detach().cpu().float().numpy()
            mtg_probs = 1.0 / (1.0 + np.exp(-mtg_logits))
            mtg_test_scores.append(mtg_probs)
            mtg_test_targets.append(labels.detach().cpu().numpy())

    # Calculate test statistics
    test_epoch_loss = test_loss / (test_deam_total + test_mtg_total) if (test_deam_total + test_mtg_total) > 0 else 0.0
    test_avg_deam_ccc = test_deam_ccc / test_deam_total if test_deam_total > 0 else 0.0

    # Log test to TensorBoard
    writer.add_scalar('Test/Loss', test_epoch_loss, args.epochs)
    writer.add_scalar('Test/DEAM_CCC', test_avg_deam_ccc, args.epochs)

    if deam_test_preds and deam_test_targets:
        deam_pred = np.concatenate(deam_test_preds, axis=0)
        deam_tgt = np.concatenate(deam_test_targets, axis=0)
        writer.add_scalar('Test/DEAM_RMSE', _rmse_np(deam_pred, deam_tgt), args.epochs)
        writer.add_scalar('Test/DEAM_Pearson', _pearson_np(deam_pred, deam_tgt), args.epochs)
        writer.add_scalar('Test/DEAM_CCC_epoch', _ccc_np(deam_pred, deam_tgt), args.epochs)

    if mtg_test_scores and mtg_test_targets:
        y_score = np.concatenate(mtg_test_scores, axis=0)
        y_true = np.concatenate(mtg_test_targets, axis=0).astype(np.int32)
        mtg_m = _mtg_metrics(y_true=y_true, y_score=y_score, threshold=0.5)
        writer.add_scalar('Test/MTG_ROC_AUC_micro', mtg_m['roc_auc_micro'], args.epochs)
        writer.add_scalar('Test/MTG_ROC_AUC_macro', mtg_m['roc_auc_macro'], args.epochs)
        writer.add_scalar('Test/MTG_PR_AUC_micro', mtg_m['pr_auc_micro'], args.epochs)
        writer.add_scalar('Test/MTG_PR_AUC_macro', mtg_m['pr_auc_macro'], args.epochs)
        writer.add_scalar('Test/MTG_F1_micro', mtg_m['f1_micro'], args.epochs)
        writer.add_scalar('Test/MTG_F1_macro', mtg_m['f1_macro'], args.epochs)
        writer.add_scalar('Test/MTG_Precision_micro', mtg_m['precision_micro'], args.epochs)
        writer.add_scalar('Test/MTG_Precision_macro', mtg_m['precision_macro'], args.epochs)
        writer.add_scalar('Test/MTG_Recall_micro', mtg_m['recall_micro'], args.epochs)
        writer.add_scalar('Test/MTG_Recall_macro', mtg_m['recall_macro'], args.epochs)

    # Print test summary
    print(f"Test Loss: {test_epoch_loss:.4f}")
    print(f"Test DEAM CCC: {test_avg_deam_ccc:.4f}")
    if deam_test_preds and deam_test_targets:
        print(f"Test DEAM RMSE: {_rmse_np(deam_pred, deam_tgt):.4f}")
        print(f"Test DEAM Pearson: {_pearson_np(deam_pred, deam_tgt):.4f}")
        print(f"Test DEAM CCC (epoch): {_ccc_np(deam_pred, deam_tgt):.4f}")
    if mtg_test_scores and mtg_test_targets:
        print(
            "Test MTG "
            f"ROC-AUC micro/macro: {mtg_m['roc_auc_micro']:.4f}/{mtg_m['roc_auc_macro']:.4f} | "
            f"PR-AUC micro/macro: {mtg_m['pr_auc_micro']:.4f}/{mtg_m['pr_auc_macro']:.4f} | "
            f"F1 micro/macro: {mtg_m['f1_micro']:.4f}/{mtg_m['f1_macro']:.4f} | "
            f"P micro/macro: {mtg_m['precision_micro']:.4f}/{mtg_m['precision_macro']:.4f} | "
            f"R micro/macro: {mtg_m['recall_micro']:.4f}/{mtg_m['recall_macro']:.4f}"
        )
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
