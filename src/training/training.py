import os, gc, logging
import torch
import pandas as pd
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    logging as transformers_logging,
)
from typing import Literal, Type
from sklearn.metrics import classification_report
from accelerate import Accelerator
from utils import Config, logger, compute_class_weights
from data import (
    load_all_datasets, 
    load_en_dataset, 
    load_id_dataset, 
    gss_split, 
    log_split_info, 
    YouTubeSpamDataset, 
    YouTubeSpamDatasetBaseline)
from models import initialize_model, create_dataloader
from .fit_model import fit_model
from evaluation import evaluate

# Training Script
def train(
        config: Config,
        dataset: Literal["all", "id", "en"] = "all",
        bert_dataset_class: Type[YouTubeSpamDataset | YouTubeSpamDatasetBaseline] = YouTubeSpamDataset
    ):
    # =========================================================================
    # 0. Inisialisasi HuggingFace Accelerator
    # =========================================================================
    accelerator = Accelerator(mixed_precision='fp16')

    # Mencegah duplikasi log di multi-GPU:
    # Hanya izinkan proses utama (GPU 0) yang mencetak log INFO,
    # sedangkan GPU 1 dsb hanya mencetak WARNING/ERROR.
    if not accelerator.is_local_main_process:
        logger.setLevel(logging.WARNING)
        transformers_logging.set_verbosity_warning()

    # Tampilkan informasi device yang digunakan
    logger.info(f"Device yang digunakan: {accelerator.device}")
    logger.info(f"Jumlah proses: {accelerator.num_processes}")
    logger.info(f"Distributed type: {accelerator.distributed_type}")
    if accelerator.device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(accelerator.device)}")
        logger.info(
            f"  Memori GPU: {torch.cuda.get_device_properties(accelerator.device).total_memory / 1e9:.1f} GB"
        )

    # =========================================================================
    # 1. Load dan preprocessing data
    # =========================================================================
    match dataset:
        case "all":
            df = load_all_datasets(config)
        case "id":
            df = load_id_dataset(config)
        case "en":
            df = load_en_dataset(config)

    logger.info(
        f"Distribusi label:\n{df['label'].value_counts().to_string()}"
    )
    logger.info(
        f"Distribusi bahasa:\n{df['language'].value_counts().to_string()}"
    )

    # =========================================================================
    # 2. Inisialisasi tokenizer BERT
    # =========================================================================
    logger.info(f"Memuat tokenizer dari '{config.MODEL_NAME}'...")
    tokenizer = BertTokenizer.from_pretrained(
        config.MODEL_NAME, model_max_length=config.MAX_LENGTH
    )

    # =========================================================================
    # Daftarkan token spesial kustom dari pipeline preprocessing
    # =========================================================================
    # Token-token ini dihasilkan oleh replace_urls(), replace_emails(), dll.
    # Mendaftarkannya sebagai add_tokens memastikan tokenizer
    # memperlakukan setiap token sebagai satu token tunggal (atomic), bukan
    # dipecah menjadi sub-word (misal: '[', 'URL', ']' → 3 token).
    CUSTOM_SPECIAL_TOKENS = ["[URL]", "[EMAIL]", "[TIMESTAMP]", "[USER]", "[HASHTAG]"]
    tokenizer.add_tokens(CUSTOM_SPECIAL_TOKENS)

    # =========================================================================
    # 3. Setup Holdout Split
    # =========================================================================
    GROUP_COL = 'video_title'


    # Membagi df untuk training dan sisanya untuk validation/testing
    df_train, df_val_test = next(
        gss_split(
            df,
            n_splits=1,
            test_size=0.3,
            group_col=GROUP_COL,
            random_state=config.RANDOM_SEED
        )
    )

    df_val, df_test = next(
        gss_split(
            df_val_test,
            n_splits=1,
            test_size=0.5,
            group_col=GROUP_COL,
            random_state=config.RANDOM_SEED
        )
    )

    # Log informasi distribusi split dan cek data leakage antar split
    log_split_info(df, df_train, df_val, df_test)

    # Buat data loader
    train_loader = create_dataloader(df_train, tokenizer, config, dataset_class=bert_dataset_class, shuffle=True)
    val_loader = create_dataloader(df_val, tokenizer, config, dataset_class=bert_dataset_class, shuffle=False)
    test_loader = create_dataloader(df_test, tokenizer, config, dataset_class=bert_dataset_class, shuffle=False)

    all_epoch_train_metrics = []

    # Buat direktori output utama (hanya di main process)
    if accelerator.is_main_process:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"MULAI TRAINING")
    logger.info("=" * 60)


    # Inisialisasi model BERT fresh (bobot pre-trained di-reset)
    model = initialize_model(config)

    # Sesuaikan ukuran embedding matrix model dengan vocab size tokenizer
    # yang sudah diperbesar oleh token-token spesial kustom.
    # resize_token_embeddings() menambahkan baris baru di embedding matrix;
    # bobot token baru diinisialisasi secara acak (mean=0, std=0.02)
    # agar sesuai dengan skala inisialisasi standar BERT.
    model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        # Inisialisasi embedding token baru dengan distribusi normal kecil
        # (sama seperti inisialisasi bobot BERT original)
        vocab_size = len(tokenizer)
        original_vocab_size = vocab_size - len(CUSTOM_SPECIAL_TOKENS)
        model.bert.embeddings.word_embeddings.weight[
            original_vocab_size:
        ].normal_(mean=0.0, std=0.02)

    # Hitung jumlah parameter
    total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(f"  Total parameter trainable: {total_params:,}")

    # Hitung class weights dari data training untuk menangani ketidakseimbangan kelas
    class_weights = compute_class_weights(df_train["label"].to_numpy())

    # =================================================================
    # Training dan Evaluasi
    # =================================================================
    model_output_dir = os.path.join(config.OUTPUT_DIR, "best_model", 'context' if bert_dataset_class == YouTubeSpamDataset else 'base')
    train_eval_result, _ = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        accelerator=accelerator,
        config=config,
        class_weights=class_weights,
        output_dir=model_output_dir,
        save_model=True,
    )

    # =================================================================
    # Evaluasi pada Test Set — menggunakan model terbaik dari training
    # =================================================================
    # Tunggu semua proses selesai sebelum memuat model terbaik
    accelerator.wait_for_everyone()

    logger.info(f"\n  Memuat model terbaik untuk evaluasi test...")
    best_model = BertForSequenceClassification.from_pretrained(model_output_dir)

    # Prepare model dan test_loader untuk evaluasi via Accelerate
    best_model, test_loader = accelerator.prepare(best_model, test_loader)

    test_metrics = evaluate(
        model=best_model,
        dataloader=test_loader,
        accelerator=accelerator,
        deskripsi="Test Final Model"
    )

    # Catat metrik train dan test
    test_result = {
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_precision_macro": test_metrics["precision_macro"],
        "test_recall_macro": test_metrics["recall_macro"],
    }
    all_epoch_train_metrics.extend(train_eval_result)

    # Log hasil test
    logger.info(f"\n  --- Test Results ---")
    logger.info(f"    Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"    Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"    Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    logger.info(f"    Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
    logger.info(f"    Test Precision (Macro): {test_metrics['precision_macro']:.4f}")
    logger.info(f"    Test Recall (Macro): {test_metrics['recall_macro']:.4f}")

    # Tampilkan confusion matrix
    cm = test_metrics["confusion_matrix"]
    logger.info(f"    Confusion Matrix:")
    logger.info(f"                 Pred Ham    Pred Spam")
    logger.info(f"    Actual Ham:    {cm[0][0]:>5}       {cm[0][1]:>5}")
    logger.info(f"    Actual Spam:   {cm[1][0]:>5}       {cm[1][1]:>5}")

    # Classification report
    report = classification_report(
        test_metrics["all_labels"],
        test_metrics["all_preds"],
        target_names=["Ham (0)", "Spam (1)"],
    )
    report_dict = classification_report(
        test_metrics["all_labels"],
        test_metrics["all_preds"],
        target_names=["Ham (0)", "Spam (1)"],
        output_dict=True,
    )
    logger.info(f"\n{report}")

    # Bebaskan memori GPU setelah evaluasi selesai
    del model, best_model, train_loader, val_loader
    del df_train, df_val, df_test

    # Bersihkan referensi internal di Accelerator (penting untuk loop)
    accelerator.free_memory()

    # Panggil garbage collector
    gc.collect()
    if accelerator.device.type == "cuda":
        torch.cuda.empty_cache()


    # =========================================================================
    # 5. Simpan Hasil ke CSV (hanya di main process)
    # =========================================================================
    if accelerator.is_main_process:
        train_results_df = pd.DataFrame(all_epoch_train_metrics)
        results_df = pd.DataFrame([test_result])
        test_result_path = os.path.join(
            config.OUTPUT_DIR,
            'test_result',
            'context' if bert_dataset_class == YouTubeSpamDataset else 'base'
        )
        os.makedirs(test_result_path, exist_ok=True)

        # Simpan hasil evaluasi training per epoch
        per_epoch_path = os.path.join(test_result_path, "train_results.csv")
        train_results_df.to_csv(per_epoch_path, index=False)

        # Simpan hasil test final
        test_results_path = os.path.join(test_result_path, "test_results.csv")
        results_df.to_csv(test_results_path, index=False)

        # Simpan classification report
        report_df = pd.DataFrame(report_dict).transpose()
        report_path = os.path.join(test_result_path, "classification_report.csv")
        report_df.to_csv(report_path, index=True)

        # Simpan confusion matrix
        cm_df = pd.DataFrame(
            cm, 
            index=["Actual Ham", "Actual Spam"], 
            columns=["Pred Ham", "Pred Spam"]
        )
        cm_path = os.path.join(test_result_path, "confusion_matrix.csv")
        cm_df.to_csv(cm_path, index=True)

        logger.info(f"\nHasil evaluasi training disimpan ke: {per_epoch_path}")
        logger.info(f"Hasil test final disimpan ke: {test_results_path}")
        logger.info(f"Classification report disimpan ke: {report_path}")
        logger.info(f"Confusion matrix disimpan ke: {cm_path}")

    logger.info("\n🎉 Training selesai!")
