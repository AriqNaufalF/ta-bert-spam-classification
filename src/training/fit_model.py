import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator
from utils import Config, logger
from .epoch import train_one_epoch
from evaluation import evaluate
from optuna import Trial, TrialPruned

def fit_model(
        model: BertForSequenceClassification,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: BertTokenizer,
        accelerator: Accelerator,
        config: Config,
        class_weights: torch.Tensor,
        output_dir: str,
        save_model: bool = True,
        trial: Trial | None = None,
    ) -> tuple[list[dict[str, int | float]], float]:
    """
    Args:
        model: Model BERT yang sudah di-inisialisasi
        train_loader: DataLoader untuk data training
        val_loader: DataLoader untuk data validasi
        tokenizer: BERT tokenizer (untuk disimpan bersama model)
        accelerator: Accelerator instance untuk multi-GPU support
        config: Objek konfigurasi
        class_weights: Bobot per kelas untuk loss function
        save_model: Apakah model perlu disimpan
        trial: Trial instance untuk optuna hyperparameter tuning (opsional)

    Returns:
        evaluation_result: Hasil evaluasi proses training
        best_f1: Nilai F1 terbaik yang dicapai selama training
    """
    if save_model and accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Setup optimizer AdamW
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Hitung total training steps dan warmup steps
    total_steps = len(train_loader) * config.EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)

    # Linear schedule with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # =========================================================================
    # Prepare semua objek training via Accelerate
    # =========================================================================
    # accelerator.prepare() secara otomatis:
    # - Memindahkan model ke device yang benar (GPU/CPU)
    # - Membungkus model dengan DistributedDataParallel jika multi-GPU
    # - Menyesuaikan DataLoader untuk distributed sampling
    # - Menyesuaikan optimizer dan scheduler
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    logger.info(f"  Total training steps: {total_steps}, Warmup: {warmup_steps}")

    # Variabel untuk menyimpan model terbaik
    best_f1 = 0.0
    best_epoch = 0
    early_stop_counter = 0
    evaluation_result = []

    for epoch in range(1, config.EPOCHS + 1):
        # Training untuk satu epoch
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            class_weights=class_weights,
            epoch=epoch,
        )
        logger.info(f"    Training Loss: {train_loss:.4f}")

        # Evaluasi pada validation set
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            accelerator=accelerator,
            class_weights=class_weights,
            deskripsi=f"Val Epoch {epoch}",
        )

        logger.info(f"    Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"    Val Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"    Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
        logger.info(f"    Val F1 (Macro): {val_metrics['f1_macro']:.4f}")
        logger.info(f"    Val Precision: {val_metrics['precision_macro']:.4f}")
        logger.info(f"    Val Recall: {val_metrics['recall_macro']:.4f}")

        # Siapkan dict untuk menyimpan hasil evaluasi
        row_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics['loss'],
            "val_accuracy": val_metrics['accuracy'],
            "val_roc_auc": val_metrics['roc_auc'],
            "val_f1_macro": val_metrics['f1_macro'],
            "val_f1_weighted": val_metrics['f1_weighted'],
            "val_precision_macro": val_metrics['precision_macro'],
            "val_recall_macro": val_metrics['recall_macro'],
        }

        evaluation_result.append(row_result)

        if trial is not None:
            # Laporkan metrik F1 Macro ke Optuna untuk pruning
            trial.report(val_metrics['f1_macro'], epoch)
            if trial.should_prune():
                logger.info(f"    Trial pruned at epoch {epoch} with F1 Macro: {val_metrics['f1_macro']:.4f}")
                accelerator.wait_for_everyone()
                raise TrialPruned()

        # Simpan model terbaik berdasarkan F1 validasi
        # Hanya simpan di main process untuk menghindari race condition
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            early_stop_counter = 0

            # Tunggu semua proses selesai sebelum menyimpan
            accelerator.wait_for_everyone()

            if save_model:
                # unwrap_model() mengeluarkan model asli dari wrapper DDP
                unwrapped_model = accelerator.unwrap_model(model)
                if accelerator.is_main_process:
                    unwrapped_model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    logger.info(
                        f"    ✅ Model terbaik disimpan! (F1 Macro: {best_f1:.4f}, Epoch: {epoch})"
                    )
                else: 
                    if accelerator.is_local_main_process:
                        logger.info(
                            f"    ✅ Model terbaik baru! (F1 Macro: {best_f1:.4f}, Epoch: {epoch})"
                        )
        else:
            early_stop_counter += 1
            logger.info(f"    ⚠️ Tidak ada peningkatan F1. Early stopping counter: {early_stop_counter}/{config.PATIENCE}")
            if early_stop_counter >= config.PATIENCE:
                logger.info("    🛑 Early stopping terpicu! Menghentikan training.")
                break

    logger.info(
        f"Best epoch: {best_epoch}, Best Val F1: {best_f1:.4f}"
    )

    return evaluation_result, best_f1
