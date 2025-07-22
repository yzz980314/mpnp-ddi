import glob
import json

import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from dataset import load_ddi_dataset
from train_logger import TrainLogger
from model import (Improved_MPNP_DDI, compute_enhanced_mpnp_loss, do_compute_metrics)
from utils import *
import torch.nn.functional as F
import time
import pandas as pd
import re
import traceback
import warnings
from utils import CustomData

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set the global font to an English font
plt.rcParams['font.family'] = 'DejaVu Sans'  # or 'Arial'

# Disable Chinese font fallback for minus sign
mpl.rcParams['axes.unicode_minus'] = False


#################################################################################################################

def enhanced_train_epoch(model, train_loader, optimizer, device, kl_weight, uncertainty_weight,
                         gradient_accumulation_steps, scaler):
    """A single training epoch for the enhanced MPNP model (final fixed version)."""
    model.train()
    running_loss = AverageMeter()
    pred_loss_meter = AverageMeter()
    kl_loss_meter = AverageMeter()
    unc_loss_meter = AverageMeter()
    all_preds = []
    all_labels = []
    optimizer.zero_grad()

    for i, batch_data in enumerate(tqdm(train_loader, desc="Training Epoch")):
        try:
            head, tail, rel, label = [d.to(device) for d in batch_data]

            # Use autocast for mixed-precision forward pass
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(head, tail, rel)
                if not (isinstance(outputs, tuple) and len(outputs) == 2):
                    continue

                predictions, uncertainty = outputs
                loss_dict = compute_enhanced_mpnp_loss(
                    predictions, uncertainty, label,
                    kl_weight=kl_weight,
                    uncertainty_weight=uncertainty_weight
                )
                loss = loss_dict['total_loss'] / gradient_accumulation_steps
                probs = loss_dict['predictions']

            # 2. Scale the loss before backpropagation
            scaler.scale(loss).backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                # 3. Use scaler.step() instead of optimizer.step()
                scaler.step(optimizer)
                # 4. Update the scaler
                scaler.update()
                optimizer.zero_grad()

            # Update metrics
            running_loss.update(loss_dict['total_loss'].item(), label.size(0))
            pred_loss_meter.update(loss_dict['prediction_loss'].item(), label.size(0))
            kl_loss_meter.update(loss_dict['kl_loss'].item(), label.size(0))
            unc_loss_meter.update(loss_dict['uncertainty_loss'].item(), label.size(0))
            all_preds.extend(probs.detach().cpu().numpy())
            all_labels.extend(label.detach().cpu().numpy())

        except Exception as e:
            print(f"Error in training batch {i}: {e}")
            traceback.print_exc()
            continue

    # Handle any remaining gradients
    if (i + 1) % gradient_accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    # Calculate epoch metrics
    if all_preds and all_labels:
        acc, auroc, f1, _, _, _, _ = do_compute_metrics(np.array(all_preds), np.array(all_labels))
    else:
        acc = auroc = f1 = 0.0

    return {
        'train_loss': running_loss.get_average(),
        'train_pred_loss': pred_loss_meter.get_average(),
        'train_kl_loss': kl_loss_meter.get_average(),
        'train_unc_loss': unc_loss_meter.get_average(),
        'train_acc': acc,
        'train_auroc': auroc,
        'train_f1': f1,
    }


def enhanced_val_epoch(model, val_loader, device, kl_weight, uncertainty_weight):
    """A single validation/test epoch for the enhanced MPNP model (fixed)."""
    model.eval()
    running_loss = AverageMeter()
    all_preds = []
    all_labels = []
    all_uncertainties = []

    with torch.no_grad():
        for batch_data in val_loader:
            try:
                head, tail, rel, label = [d.to(device) for d in batch_data]

                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(head, tail, rel)
                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        predictions, uncertainty = outputs
                    else:
                        continue

                    loss_dict = compute_enhanced_mpnp_loss(
                        predictions, uncertainty, label,
                        kl_weight=kl_weight,
                        uncertainty_weight=uncertainty_weight
                    )
                    loss = loss_dict['total_loss']
                    probs = loss_dict['predictions']

                running_loss.update(loss.item(), label.size(0))
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                all_uncertainties.extend(uncertainty.cpu().numpy())

            except Exception as e:
                print(f"Error during validation/test batch: {e}")
                continue

    # ==================== Key fix is here ====================
    # Ensure all 7 metrics are correctly received and returned
    if all_preds and all_labels:
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc, auroc, f1, precision, recall, ap, aupr = do_compute_metrics(all_preds, all_labels)
    else:
        acc = auroc = f1 = precision = recall = ap = aupr = 0.0
    # =======================================================

    # Calculate uncertainty-related metrics
    all_uncertainties = np.array(all_uncertainties)
    avg_uncertainty = all_uncertainties.mean()
    pred_errors = (all_preds - all_labels) ** 2
    # Calculate the correlation between uncertainty and error
    if len(all_uncertainties) > 1 and len(pred_errors) > 1:
        unc_err_corr = np.corrcoef(all_uncertainties, pred_errors)[0, 1]
    else:
        unc_err_corr = 0.0

    return {
        'val_loss': running_loss.get_average(),
        'val_acc': acc,
        'val_auroc': auroc,
        'val_f1': f1,
        'val_precision': precision,
        'val_recall': recall,
        'val_ap': ap,
        'val_aupr': aupr,  # <--- Added the missing aupr
        'val_unc': avg_uncertainty,
        'val_unc_err_corr': unc_err_corr if not np.isnan(unc_err_corr) else 0.0
    }


def enhanced_test_epoch(model, test_loader, device, epoch, logger):
    """A single test epoch for the enhanced training function (fixed)."""
    model.eval()
    running_loss = AverageMeter()
    all_preds = []
    all_labels = []
    all_uncs = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            head, tail, rel, label = [d.to(device) for d in batch_data]

            try:
                # Process model output
                outputs = model(head, tail, rel)

                # Determine model output type
                if isinstance(outputs, tuple):
                    # ==================== Main change is here ====================
                    # The model now returns (scores, uncertainty), so unpack into 2 values
                    predictions, uncertainty = outputs
                    # =======================================================

                    # Standardize predictions (logits)
                    if predictions.dim() > 1:
                        predictions = predictions.squeeze(-1)

                    # Standardize labels
                    if label.dim() > 1:
                        label = label.squeeze(-1)

                    # Use logits to calculate binary classification loss
                    loss = F.binary_cross_entropy_with_logits(predictions, label)
                    probs = torch.sigmoid(predictions)

                    # Process uncertainty
                    if uncertainty is not None:
                        all_uncs.extend(uncertainty.cpu().numpy())
                    else:
                        all_uncs.extend(np.zeros(predictions.size(0)))

                else:  # Maintain compatibility with simplified models
                    predictions = outputs.squeeze(-1)
                    if predictions.dim() > 1:
                        predictions = predictions.squeeze(-1)
                    if label.dim() > 1:
                        label = label.squeeze(-1)
                    loss = F.binary_cross_entropy_with_logits(predictions, label)
                    probs = torch.sigmoid(predictions)
                    all_uncs.extend(np.zeros(predictions.size(0)))

                running_loss.update(loss.item(), label.size(0))
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

            except Exception as e:
                print(f"Error in test batch {batch_idx}: {e}")
                traceback.print_exc()  # Print detailed error stack
                continue

    # Calculate evaluation metrics
    pred_probs = np.array(all_preds)
    labels = np.array(all_labels)
    uncertainties = np.array(all_uncs)

    if len(pred_probs) == 0:
        print("Warning: No valid predictions in the test set, cannot compute metrics.")
        return {
            'test_loss': running_loss.get_average(), 'test_acc': 0, 'test_auroc': 0,
            'test_f1': 0, 'test_precision': 0, 'test_recall': 0, 'test_ap': 0,
            'test_unc': 0, 'test_unc_err_corr': 0
        }

    pred_classes = (pred_probs > 0.5).astype(int)
    acc = accuracy_score(labels, pred_classes)

    if len(np.unique(labels)) > 1:
        auroc = roc_auc_score(labels, pred_probs)
        f1 = f1_score(labels, pred_classes)
        precision = precision_score(labels, pred_classes)
        recall = recall_score(labels, pred_classes)
        ap = average_precision_score(labels, pred_probs)
    else:
        auroc = f1 = precision = recall = ap = 0.0

    errors = np.abs(pred_probs - labels)
    unc_err_corr = 0.0
    if len(uncertainties) > 1 and np.std(uncertainties) > 1e-9 and np.std(errors) > 1e-9:
        try:
            unc_err_corr = np.corrcoef(uncertainties, errors)[0, 1]
            if np.isnan(unc_err_corr):
                unc_err_corr = 0.0
        except Exception as corr_e:
            print(f"Error computing uncertainty correlation: {corr_e}")
            unc_err_corr = 0.0

    print(f"Epoch {epoch + 1} [Test]: Loss: {running_loss.get_average():.4f}, "
          f"Acc: {acc:.4f}, AUROC: {auroc:.4f}, F1: {f1:.4f}")

    return {
        'test_loss': running_loss.get_average(),
        'test_acc': acc,
        'test_auroc': auroc,
        'test_f1': f1,
        'test_precision': precision,
        'test_recall': recall,
        'test_ap': ap,
        'test_unc': np.mean(uncertainties) if len(uncertainties) > 0 else 0,
        'test_unc_err_corr': unc_err_corr
    }


def enhanced_train_mpnp(model, optimizer, scheduler, train_loader, val_loader, test_loader, device, epochs, logger,
                        kl_weight, uncertainty_weight, save_model, patience, gradient_accumulation_steps,
                        max_batch_size):
    """Complete main training function for the enhanced MPNP model (final fixed version)."""
    best_val_auroc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    # GradScaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(epochs):
        print(f"\n{'=' * 50}\nEpoch {epoch + 1}/{epochs}\n{'=' * 50}")

        # ==================== Key fix here: Call with keyword arguments ====================
        # This ensures that each parameter is passed to the correct position, avoiding confusion.
        train_metrics = enhanced_train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            kl_weight=kl_weight,
            uncertainty_weight=uncertainty_weight,
            gradient_accumulation_steps=gradient_accumulation_steps,
            scaler=scaler
        )
        # =======================================================================

        # The calls for validation and testing are correct as they have fewer parameters and are less prone to confusion.
        val_metrics = enhanced_val_epoch(
            model=model,
            val_loader=val_loader,
            device=device,
            kl_weight=kl_weight,
            uncertainty_weight=uncertainty_weight
        )
        test_metrics = enhanced_val_epoch(
            model=model,
            val_loader=test_loader,
            device=device,
            kl_weight=kl_weight,
            uncertainty_weight=uncertainty_weight
        )
        test_metrics_renamed = {f"test_{k.split('_', 1)[1]}": v for k, v in test_metrics.items()}

        scheduler.step()

        all_metrics = {**train_metrics, **val_metrics, **test_metrics_renamed}
        logger.record_metrics(epoch, all_metrics)
        print(f"Epoch {epoch} completed.")

        current_val_auroc = all_metrics.get('val_auroc', 0.0)
        if current_val_auroc > best_val_auroc + 0.001:
            best_val_auroc = current_val_auroc
            best_epoch = epoch
            epochs_no_improve = 0
            if save_model:
                model_path = os.path.join(logger.get_model_dir(), 'best_model.pth')
                torch.save(model.state_dict(), model_path)
                logger.info(f"Best model saved (epoch {epoch + 1}, val_auroc: {current_val_auroc:.4f})")
        else:
            epochs_no_improve += 1
            logger.info(f"Validation AUROC did not improve. Waiting {epochs_no_improve}/{patience} epochs.")
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered.")
                break

    print(f"\nTraining complete! Best validation AUROC: {best_val_auroc:.4f} (Epoch {best_epoch + 1})")

    if save_model and os.path.exists(os.path.join(logger.get_model_dir(), 'best_model.pth')):
        print("Loading the best model for final testing...")
        model.load_state_dict(torch.load(os.path.join(logger.get_model_dir(), 'best_model.pth')))
        final_test_metrics = enhanced_val_epoch(model, test_loader, device, kl_weight, uncertainty_weight)
        print(
            f"Final performance of the best model on the test set: AUROC={final_test_metrics['val_auroc']:.4f}, F1={final_test_metrics['val_f1']:.4f}")

    return best_val_auroc, best_epoch, logger


###########################################################################################################
def plot_training_from_history(history, save_path=None):
    """
   Create and plot training curves directly from an in-memory history (a list of dictionaries).

   Args:
       history (list): A list where each element is a dictionary containing the metrics for that epoch.
       save_path (str, optional): The path to save the chart.
   """
    if not history:
        print("Warning: Training history is empty, cannot generate chart.")
        return

    # Convert the history to a pandas DataFrame, which makes data handling very convenient.
    df = pd.DataFrame(history)

    # Ensure the epoch column exists.
    if 'epoch' not in df.columns:
        df['epoch'] = range(len(df))

    # Create multiple subplots.
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Training Process Analysis (from memory)', fontsize=16)

    # 1. Overall loss curve
    ax1 = axes[0, 0]
    if 'train_loss' in df.columns:
        ax1.plot(df['epoch'], df['train_loss'], 'b-', marker='o', label='Training Loss')
    if 'val_loss' in df.columns:
        ax1.plot(df['epoch'], df['val_loss'], 'r-', marker='s', label='Validation Loss')
    ax1.set_title('Total Loss Trend')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 2. Comparison of various loss components
    ax2 = axes[0, 1]
    if 'train_pred_loss' in df.columns:
        ax2.plot(df['epoch'], df['train_pred_loss'], 'g-', marker='o', label='Prediction Loss')
    if 'train_kl_loss' in df.columns:
        ax2.plot(df['epoch'], df['train_kl_loss'], 'm-', marker='s', label='KL Divergence Loss')
    if 'train_unc_loss' in df.columns:
        ax2.plot(df['epoch'], df['train_unc_loss'], 'c-', marker='^', label='Uncertainty Loss')
    ax2.set_title('Loss Component Trends')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 3. AUROC comparison
    ax3 = axes[1, 0]
    if 'train_auroc' in df.columns:
        ax3.plot(df['epoch'], df['train_auroc'], 'b-', marker='o', label='Train AUROC')
    if 'val_auroc' in df.columns:
        ax3.plot(df['epoch'], df['val_auroc'], 'r-', marker='s', label='Validation AUROC')
    if 'test_auroc' in df.columns:
        ax3.plot(df['epoch'], df['test_auroc'], 'g-', marker='^', label='Test AUROC')
    ax3.axhline(y=0.5, color='k', linestyle='--', label='Random Guess')
    ax3.set_title('AUROC Trend')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('AUROC')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)

    # 4. Training vs. Validation loss comparison
    ax4 = axes[1, 1]
    if 'train_loss' in df.columns and 'val_loss' in df.columns:
        ax4.scatter(df['train_loss'], df['val_loss'], c=df['epoch'], cmap='viridis', s=100, alpha=0.8)
        ax4.set_title('Training vs. Validation Loss')
        ax4.set_xlabel('Training Loss')
        ax4.set_ylabel('Validation Loss')
        ax4.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    else:
        plt.show()  # If not saving, just display it.


def check_data_batch(batch_data, verbose=True):
    """Check the format and content of a data batch."""
    try:
        if len(batch_data) != 4:
            print(f"Warning: Batch data length is {len(batch_data)}, expected 4")
            return False

        h_data, t_data, rel, label = batch_data

        # Check basic attributes of head and tail graphs
        if not hasattr(h_data, 'x') or not hasattr(t_data, 'x'):
            print("Error: Graph data is missing node features 'x'")
            return False

        if not hasattr(h_data, 'edge_index') or not hasattr(t_data, 'edge_index'):
            print("Error: Graph data is missing edge index 'edge_index'")
            return False

        if not hasattr(h_data, 'edge_attr') or not hasattr(t_data, 'edge_attr'):
            print("Error: Graph data is missing edge attributes 'edge_attr'")
            return False

        if verbose:
            print("\n=== Data Batch Check ===")
            print(
                f"Head graph: Nodes={h_data.x.size(0)}, Feature dim={h_data.x.size(1)}, Edges={h_data.edge_index.size(1)}")
            print(
                f"Tail graph: Nodes={t_data.x.size(0)}, Feature dim={t_data.x.size(1)}, Edges={t_data.edge_index.size(1)}")
            print(f"Relations: Shape={rel.shape}, Range=[{rel.min().item()}, {rel.max().item()}]")
            print(f"Labels: Shape={label.shape}, Range=[{label.min().item()}, {label.max().item()}]")
            print(
                f"Label distribution: Positive={torch.sum(label > 0.5).item()}, Negative={torch.sum(label <= 0.5).item()}")

            # Check statistics of node and edge features
            print(f"Head graph node features: Mean={h_data.x.mean().item():.4f}, Std={h_data.x.std().item():.4f}")
            print(f"Tail graph node features: Mean={t_data.x.mean().item():.4f}, Std={t_data.x.std().item():.4f}")
            print(
                f"Head graph edge features: Mean={h_data.edge_attr.mean().item():.4f}, Std={h_data.edge_attr.std().item():.4f}")
            print(
                f"Tail graph edge features: Mean={t_data.edge_attr.mean().item():.4f}, Std={t_data.edge_attr.std().item():.4f}")

        return True

    except Exception as e:
        print(f"Error while checking data batch: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Print GPU memory information
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.cuda.empty_cache()

    # --- Model Configuration ---
    config = {
        # ==================== Key Change 1: Define the list of models to train ====================
        'model_types_to_train': ['enhanced_mpnp'],
        # ==========================================================================================
        'n_iter': 2, 'fold': 0, 'weight_decay': 5e-4, 'save_model': True,
        'lr': 0.0001, 'kge_dim': 32, 'rel_total': 255,  # drugbank:86,
        'kl_weight': 0.001,
        'uncertainty_weight': 0.001,
        'data_root': '/home/yanzimo/project/DGNN-DDI-main/data/preprocessed/',
        'dataset_name': 'decagon',  # 'drugbank',
        'save_dir_base': 'save', 'subset_size': -1,
        'latent_dim': 16, 'memory_size': 50, 'max_iterations': 3,
        'epochs': 20, 'batch_size': 32, 'hidden_dim': 32,
        'use_preprocessed_data': True, 'preprocessed_train_file': 'decagon1_processed_train_fold{}.csv',
        'preprocessed_test_file': 'decagon_processed_test_fold{}.csv',
        'gradient_accumulation_steps': 4, 'max_batch_size': 8, 'patience': 5
    }
    print(f"Model Configuration: {config}")

    all_experiment_results = {}

    # Load dataset
    data_path = os.path.join(config['data_root'], config['dataset_name'])
    print(f"Loading dataset from: {data_path}, using the first {config['subset_size']} records...")
    # New call in train.py (with two added parameters)
    train_loader, val_loader, test_loader = load_ddi_dataset(
        root=data_path,
        batch_size=config['batch_size'],
        fold=config['fold'],
        subset_size=config['subset_size'],
        augment_data=False,
        # Pass the filename templates from the config
        train_file_template=config['preprocessed_train_file'],
        test_file_template=config['preprocessed_test_file']
    )

    print(f"Training set samples: {len(train_loader.dataset) if train_loader else 0}")
    print(f"Validation set samples: {len(val_loader.dataset) if val_loader else 0}")
    print(f"Test set samples: {len(test_loader.dataset) if test_loader else 0}")

    # Get data dimensions
    try:
        first_batch = next(iter(train_loader))
        node_dim = first_batch[0].x.size(-1)
        edge_dim = first_batch[0].edge_attr.size(-1) if first_batch[0].edge_attr is not None else 1
    except Exception as e:
        print(f"Error getting data dimensions from DataLoader: {e}, {traceback.format_exc()}")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Node dim: {node_dim}, Edge dim: {edge_dim}, Using device: {device}")

    # Create top-level experiment save directory
    current_time_str = time.strftime("%Y%m%d_%H%M%S")
    experiment_save_dir = os.path.join(config['save_dir_base'],
                                       f"{current_time_str}_{config['dataset_name']}_experiments")
    config['experiment_save_dir'] = experiment_save_dir
    os.makedirs(experiment_save_dir, exist_ok=True)
    print(f"All experiment results will be saved in: {experiment_save_dir}")

    # ==================== Key Change 2: Loop through the list of models ====================
    for model_type_to_train in config['model_types_to_train']:
        print(f"\n{'=' * 60}\nStarting training for {model_type_to_train.upper()} model\n{'=' * 60}")

        # The config passed to the logger needs to include the current model type to create the correct directory
        logger_config = config.copy()
        logger_config['model_types_to_train'] = [model_type_to_train]
        logger = TrainLogger(logger_config)
        logger.info(f"Training {model_type_to_train.upper()} model")
        logger.info(f"Parameters: {json.dumps(config, indent=4)}")

        # Create model
        model = None
        if model_type_to_train == 'enhanced_mpnp':
            config['hidden_dim'] = config['kge_dim']
            print(f"Using unified dimension: hidden_dim = kge_dim = {config['hidden_dim']}")
            model = Improved_MPNP_DDI(
                in_dim=node_dim, edge_dim=edge_dim, hidden_dim=config['hidden_dim'],
                n_iter=config['n_iter'], kge_dim=config['kge_dim'], rel_total=config['rel_total']
            )
            print("Successfully created the enhanced model")
        else:
            print(f"Error: Unknown model type '{model_type_to_train}'")
            continue

        model.to(device=device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model total parameters: {total_params / 1e6:.2f}M")

        # Optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['lr'] * 0.01)

        # Call the appropriate main training function based on model type
        try:
            if model_type_to_train == 'enhanced_mpnp':
                best_auroc, best_epoch, logger = enhanced_train_mpnp(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                    device=device, epochs=config['epochs'], logger=logger,
                    kl_weight=config['kl_weight'], uncertainty_weight=config['uncertainty_weight'],
                    save_model=config['save_model'], patience=config['patience'],
                    gradient_accumulation_steps=config['gradient_accumulation_steps'],
                    max_batch_size=config['max_batch_size']
                )
            print(f"Training complete! Best validation AUROC: {best_auroc:.4f} (Epoch {best_epoch + 1})")
        except Exception as e:
            print(f"A critical error occurred during training: {e}, {traceback.format_exc()}")
            continue

        # Plotting logic (now generic)
        print(f"\nGenerating performance charts for {model_type_to_train.upper()}...")
        history = logger.get_performance_metrics()
        if history:
            chart_save_path = os.path.join(logger.train_save_dir, f"{model_type_to_train}_training_curves.png")
            plot_training_from_history(history, chart_save_path)
            all_experiment_results[model_type_to_train] = pd.DataFrame(history)
        else:
            print("Warning: Training history is empty, cannot generate chart.")

    # ==================== Key Change 3: Add final performance comparison ====================
    print(f"\n{'=' * 60}\nAll models trained, final performance comparison:\n{'=' * 60}")
    summary_list = []
    for model_name, results_df in all_experiment_results.items():
        if not results_df.empty and 'val_auroc' in results_df.columns:
            best_epoch_idx = results_df['val_auroc'].idxmax()
            best_metrics = results_df.loc[best_epoch_idx]
            summary = {
                'Model': model_name,
                'Best Epoch': int(best_metrics.get('epoch', -1)) + 1,
                'Val AUROC': best_metrics.get('val_auroc', 0.0),
                'Test AUROC': best_metrics.get('test_auroc', 0.0),
                'Test F1': best_metrics.get('test_f1', 0.0),
                'Test AUPR': best_metrics.get('test_aupr', 0.0)
            }
            summary_list.append(summary)

    if summary_list:
        summary_df = pd.DataFrame(summary_list).set_index('Model')
        print(summary_df)
        summary_df.to_csv(os.path.join(experiment_save_dir, 'final_summary.csv'))
        print(f"\nFinal comparison report saved to: {os.path.join(experiment_save_dir, 'final_summary.csv')}")
    else:
        print("No valid experiment results to compare.")

    print("\nExperiment script execution finished.")


if __name__ == "__main__":
    main()
