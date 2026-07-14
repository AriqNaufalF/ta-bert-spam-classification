import os, gc, logging, shutil
import torch
import pandas as pd
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    logging as transformers_logging,
)
from typing import Any, Literal, Type, cast
from sklearn.metrics import classification_report
from accelerate import Accelerator
from utils import Config, logger, compute_class_weights
from data import (
    load_all_datasets, 
    load_en_dataset, 
    load_id_dataset, 
    gss_split, 
    group_kfold_split,
    log_split_info, 
    log_cv_split_info,
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

    # Ablation test: ganti context test set dengan context acak dari train set
    # Hanya dilakukan jika menggunakan model context (YouTubeSpamDataset)
    test_ablation_loader = None
    if bert_dataset_class == YouTubeSpamDataset:
        df_test_ablation = df_test.copy()

        # Ambil daftar context unik dari training set
        context_list = df_train[['video_title', 'video_description']].drop_duplicates()
        logger.info(f"Jumlah data context unik di train set: {len(context_list)}")

        # Tukar context di test ablation set dengan context acak dari train set
        sampled_contexts = context_list.sample(
            n=len(df_test_ablation), replace=True, random_state=config.RANDOM_SEED
        ).reset_index(drop=True)
        df_test_ablation = df_test_ablation.reset_index(drop=True)
        df_test_ablation['video_title'] = sampled_contexts['video_title'].values
        df_test_ablation['video_description'] = sampled_contexts['video_description'].values

        # Log contoh pertukaran context
        logger.info(f"Contoh pertukaran context di test ablation set:")
        logger.info(f"  Original context: {df_test.iloc[0]['video_title']} | {df_test.iloc[0]['comment']}")
        logger.info(f"  Ablation context: {df_test_ablation.iloc[0]['video_title']} | {df_test_ablation.iloc[0]['comment']}")

        test_ablation_loader = create_dataloader(df_test_ablation, tokenizer, config, dataset_class=bert_dataset_class, shuffle=False)

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

    # =================================================================
    # Ablation Test — evaluasi model dengan context yang ditukar
    # Hanya dilakukan jika menggunakan model context (YouTubeSpamDataset)
    # =================================================================
    ablation_test_result = None
    ablation_report_dict = None
    ablation_cm = None
    if bert_dataset_class == YouTubeSpamDataset and test_ablation_loader is not None:
        logger.info(f"\n  Memulai evaluasi ablation test (context ditukar)...")
        best_model, test_ablation_loader = accelerator.prepare(best_model, test_ablation_loader)

        ablation_metrics = evaluate(
            model=best_model,
            dataloader=test_ablation_loader,
            accelerator=accelerator,
            deskripsi="Test Ablation (Context Swap)"
        )

        ablation_test_result = {
            "test_ablation_loss": ablation_metrics["loss"],
            "test_ablation_accuracy": ablation_metrics["accuracy"],
            "test_ablation_roc_auc": ablation_metrics["roc_auc"],
            "test_ablation_f1_macro": ablation_metrics["f1_macro"],
            "test_ablation_precision_macro": ablation_metrics["precision_macro"],
            "test_ablation_recall_macro": ablation_metrics["recall_macro"],
        }

        # Log hasil ablation test
        logger.info(f"\n  --- Ablation Test Results (Context Swap) ---")
        logger.info(f"    Ablation Test Loss: {ablation_metrics['loss']:.4f}")
        logger.info(f"    Ablation Test Accuracy: {ablation_metrics['accuracy']:.4f}")
        logger.info(f"    Ablation Test ROC-AUC: {ablation_metrics['roc_auc']:.4f}")
        logger.info(f"    Ablation Test F1 (Macro): {ablation_metrics['f1_macro']:.4f}")
        logger.info(f"    Ablation Test Precision (Macro): {ablation_metrics['precision_macro']:.4f}")
        logger.info(f"    Ablation Test Recall (Macro): {ablation_metrics['recall_macro']:.4f}")

        # Confusion matrix ablation
        ablation_cm = ablation_metrics["confusion_matrix"]
        logger.info(f"    Confusion Matrix (Ablation):")
        logger.info(f"                 Pred Ham    Pred Spam")
        logger.info(f"    Actual Ham:    {ablation_cm[0][0]:>5}       {ablation_cm[0][1]:>5}")
        logger.info(f"    Actual Spam:   {ablation_cm[1][0]:>5}       {ablation_cm[1][1]:>5}")

        # Classification report ablation
        ablation_report = classification_report(
            ablation_metrics["all_labels"],
            ablation_metrics["all_preds"],
            target_names=["Ham (0)", "Spam (1)"],
        )
        ablation_report_dict = classification_report(
            ablation_metrics["all_labels"],
            ablation_metrics["all_preds"],
            target_names=["Ham (0)", "Spam (1)"],
            output_dict=True,
        )
        logger.info(f"\n{ablation_report}")

        # Perbandingan metrik normal vs ablation
        logger.info(f"\n  --- Perbandingan Normal vs Ablation ---")
        logger.info(f"    Accuracy:  {test_metrics['accuracy']:.4f} → {ablation_metrics['accuracy']:.4f} (Δ={ablation_metrics['accuracy'] - test_metrics['accuracy']:+.4f})")
        logger.info(f"    F1 Macro:  {test_metrics['f1_macro']:.4f} → {ablation_metrics['f1_macro']:.4f} (Δ={ablation_metrics['f1_macro'] - test_metrics['f1_macro']:+.4f})")
        logger.info(f"    ROC-AUC:   {test_metrics['roc_auc']:.4f} → {ablation_metrics['roc_auc']:.4f} (Δ={ablation_metrics['roc_auc'] - test_metrics['roc_auc']:+.4f})")

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

        # Simpan hasil ablation test (hanya jika ada)
        if ablation_test_result is not None:
            # Simpan metrik ablation test
            ablation_results_df = pd.DataFrame([ablation_test_result])
            ablation_results_path = os.path.join(test_result_path, "ablation_test_results.csv")
            ablation_results_df.to_csv(ablation_results_path, index=False)

            # Simpan classification report ablation
            ablation_report_df = pd.DataFrame(ablation_report_dict).transpose()
            ablation_report_path = os.path.join(test_result_path, "ablation_classification_report.csv")
            ablation_report_df.to_csv(ablation_report_path, index=True)

            # Simpan confusion matrix ablation
            ablation_cm_df = pd.DataFrame(
                ablation_cm,
                index=["Actual Ham", "Actual Spam"],
                columns=["Pred Ham", "Pred Spam"]
            )
            ablation_cm_path = os.path.join(test_result_path, "ablation_confusion_matrix.csv")
            ablation_cm_df.to_csv(ablation_cm_path, index=True)

            logger.info(f"Hasil ablation test disimpan ke: {ablation_results_path}")
            logger.info(f"Ablation classification report disimpan ke: {ablation_report_path}")
            logger.info(f"Ablation confusion matrix disimpan ke: {ablation_cm_path}")

    logger.info("\n🎉 Training selesai!")


# =============================================================================
# Group Shuffle Split Cross-Validation (GSS-CV)
# =============================================================================
def train_gss_cv(
        config: Config,
        dataset: Literal["all", "id", "en"] = "all",
        bert_dataset_class: Type[YouTubeSpamDataset | YouTubeSpamDatasetBaseline] = YouTubeSpamDataset
    ):
    """
    Group Shuffle Split Cross-Validation.
    Membagi dataset menjadi 5 fold menggunakan GroupShuffleSplit (berdasarkan video_title).
    Model di-inisialisasi ulang dari awal untuk setiap fold.
    Hanya menyimpan metrik evaluasi (tidak menyimpan model).

    Args:
        config: Objek konfigurasi
        dataset: Dataset yang digunakan ('all', 'id', atau 'en')
        bert_dataset_class: Kelas Dataset (YouTubeSpamDataset atau YouTubeSpamDatasetBaseline)
    """
    # =========================================================================
    # 0. Inisialisasi HuggingFace Accelerator
    # =========================================================================
    accelerator = Accelerator(mixed_precision='fp16')

    # Mencegah duplikasi log di multi-GPU
    if not accelerator.is_local_main_process:
        logger.setLevel(logging.WARNING)
        transformers_logging.set_verbosity_warning()

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

    CUSTOM_SPECIAL_TOKENS = ["[URL]", "[EMAIL]", "[TIMESTAMP]", "[USER]", "[HASHTAG]"]
    tokenizer.add_tokens(CUSTOM_SPECIAL_TOKENS)

    # =========================================================================
    # 3. Group Shuffle Split Cross-Validation
    # =========================================================================
    GROUP_COL = 'video_title'
    n_splits = config.N_SPLITS

    # Buat direktori output (hanya di main process)
    cv_result_path = os.path.join(
        config.OUTPUT_DIR,
        'cv_result',
        'context' if bert_dataset_class == YouTubeSpamDataset else 'base'
    )
    if accelerator.is_main_process:
        os.makedirs(cv_result_path, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"MULAI GSS-CV ({n_splits} fold)")
    logger.info("=" * 60)

    # List untuk menyimpan hasil per fold
    all_fold_test_metrics = []
    all_fold_train_metrics = []
    all_fold_classification_reports = []
    all_fold_ablation_test_metrics = []
    all_fold_ablation_classification_reports = []

    for fold, (df_train_val, df_test) in enumerate(group_kfold_split(df, n_splits=n_splits, group_col=GROUP_COL), start=1):
        # Identifikasi grup yang dijadikan test set pada fold ini
        test_groups = sorted(df_test[GROUP_COL].unique().tolist())
        group_name = ", ".join(str(g) for g in test_groups)
        logger.info("=" * 60)
        logger.info(f"FOLD {fold}/{n_splits} — Held-out group(s): '{group_name}'")
        logger.info("=" * 60)

        # =================================================================
        # 3a. Split train_val menjadi train dan val
        # =================================================================
        df_train, df_val = next(
            gss_split(
                df_train_val,
                n_splits=1,
                test_size=0.15,
                group_col=GROUP_COL,
                random_state=config.RANDOM_SEED + fold
            )
        )

        # Log informasi split
        log_cv_split_info(df, df_train, df_val, df_test, fold, group_name)

        # Buat data loaders
        train_loader = create_dataloader(df_train, tokenizer, config, dataset_class=bert_dataset_class, shuffle=True)
        val_loader = create_dataloader(df_val, tokenizer, config, dataset_class=bert_dataset_class, shuffle=False)
        test_loader = create_dataloader(df_test, tokenizer, config, dataset_class=bert_dataset_class, shuffle=False)

        df_test_ablation = None
        test_ablation_loader = None
        if bert_dataset_class == YouTubeSpamDataset:
            # Inisiasi test ablation set
            df_test_ablation = df_test.copy()

            # Mengambil list unik data context dari df_train
            context_list = df_train[['video_title', 'video_description']].drop_duplicates()

            logger.info(f"  Jumlah data context unik di train set: {len(context_list)}")

            # Menukar secara acak salah satu context di context_list untuk setiap baris di df_test_ablation
            sampled_contexts = context_list.sample(
                n=len(df_test_ablation), replace=True, random_state=config.RANDOM_SEED
            ).reset_index(drop=True)
            df_test_ablation = df_test_ablation.reset_index(drop=True)
            df_test_ablation['video_title'] = sampled_contexts['video_title'].values
            df_test_ablation['video_description'] = sampled_contexts['video_description'].values

            # Log contoh satu komentar sebelum dan sesudah pertukaran context di test ablation set
            logger.info(f"  Contoh pertukaran context di test ablation set:")
            logger.info(f"    Original context: {df_test.iloc[0]['video_title']} | {df_test.iloc[0]['comment']}")
            logger.info(f"    Ablation context: {df_test_ablation.iloc[0]['video_title']} | {df_test_ablation.iloc[0]['comment']}")

            test_ablation_loader = create_dataloader(df_test_ablation, tokenizer, config, dataset_class=bert_dataset_class, shuffle=False)


        # =================================================================
        # 3b. Inisialisasi model BERT fresh untuk setiap fold
        # =================================================================
        model = initialize_model(config)
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            vocab_size = len(tokenizer)
            original_vocab_size = vocab_size - len(CUSTOM_SPECIAL_TOKENS)
            model.bert.embeddings.word_embeddings.weight[
                original_vocab_size:
            ].normal_(mean=0.0, std=0.02)

        total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        logger.info(f"  Total parameter trainable: {total_params:,}")

        # Hitung class weights dari data training fold ini
        class_weights = compute_class_weights(df_train["label"].to_numpy())

        # =================================================================
        # 3c. Training — simpan model sementara untuk reload best model
        # =================================================================
        # Gunakan direktori sementara untuk menyimpan best model per fold
        fold_tmp_dir = os.path.join(cv_result_path, f"_tmp_fold_{fold}")
        if accelerator.is_main_process:
            os.makedirs(fold_tmp_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        train_eval_result, _ = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            accelerator=accelerator,
            config=config,
            class_weights=class_weights,
            output_dir=fold_tmp_dir,
            save_model=True,
        )

        # Tambahkan kolom fold ke setiap baris hasil training
        for row in train_eval_result:
            metric_row = cast(dict[str, Any], row)
            metric_row["fold"] = fold
            metric_row["held_out_group"] = group_name
        all_fold_train_metrics.extend(train_eval_result)

        # =================================================================
        # 3d. Evaluasi pada Test Set (held-out group)
        # =================================================================
        accelerator.wait_for_everyone()

        logger.info(f"\n  Memuat model terbaik fold {fold} untuk evaluasi test...")
        best_model = BertForSequenceClassification.from_pretrained(fold_tmp_dir)
        best_model, test_loader = accelerator.prepare(best_model, test_loader)

        test_metrics = evaluate(
            model=best_model,
            dataloader=test_loader,
            accelerator=accelerator,
            deskripsi=f"Test Fold {fold}"
        )

        # Log hasil test fold
        logger.info(f"\n  --- Test Results Fold {fold} ---")
        logger.info(f"    Test Loss: {test_metrics['loss']:.4f}")
        logger.info(f"    Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"    Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"    Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
        logger.info(f"    Test Precision (Macro): {test_metrics['precision_macro']:.4f}")
        logger.info(f"    Test Recall (Macro): {test_metrics['recall_macro']:.4f}")

        # Confusion matrix
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

        # Evaluasi pada Test Ablation Set (held-out group) jika model menggunakan context (YouTubeSpamDataset)
        if bert_dataset_class == YouTubeSpamDataset:
            logger.info(f"\n  Memuat model terbaik fold {fold} untuk evaluasi test ablation set...")
            best_model, test_ablation_loader = accelerator.prepare(best_model, test_ablation_loader)

            test_ablation_metrics = evaluate(
                model=best_model,
                dataloader=test_ablation_loader,
                accelerator=accelerator,
                deskripsi=f"Test Ablation Fold {fold}"
            )

            # Log hasil ablation test fold
            logger.info(f"\n  --- Ablation Test Results Fold {fold} (Context Swap) ---")
            logger.info(f"    Ablation Test Loss: {test_ablation_metrics['loss']:.4f}")
            logger.info(f"    Ablation Test Accuracy: {test_ablation_metrics['accuracy']:.4f}")
            logger.info(f"    Ablation Test ROC-AUC: {test_ablation_metrics['roc_auc']:.4f}")
            logger.info(f"    Ablation Test F1 (Macro): {test_ablation_metrics['f1_macro']:.4f}")
            logger.info(f"    Ablation Test Precision (Macro): {test_ablation_metrics['precision_macro']:.4f}")
            logger.info(f"    Ablation Test Recall (Macro): {test_ablation_metrics['recall_macro']:.4f}")

            # Confusion matrix ablation
            ablation_cm = test_ablation_metrics["confusion_matrix"]
            logger.info(f"    Confusion Matrix (Ablation):")
            logger.info(f"                 Pred Ham    Pred Spam")
            logger.info(f"    Actual Ham:    {ablation_cm[0][0]:>5}       {ablation_cm[0][1]:>5}")
            logger.info(f"    Actual Spam:   {ablation_cm[1][0]:>5}       {ablation_cm[1][1]:>5}")

            # Classification report ablation
            ablation_report = classification_report(
                test_ablation_metrics["all_labels"],
                test_ablation_metrics["all_preds"],
                target_names=["Ham (0)", "Spam (1)"],
            )
            ablation_report_dict = classification_report(
                test_ablation_metrics["all_labels"],
                test_ablation_metrics["all_preds"],
                target_names=["Ham (0)", "Spam (1)"],
                output_dict=True,
            )
            logger.info(f"\n{ablation_report}")

            # Perbandingan metrik normal vs ablation per fold
            logger.info(f"\n  --- Perbandingan Normal vs Ablation (Fold {fold}) ---")
            logger.info(f"    Accuracy:  {test_metrics['accuracy']:.4f} → {test_ablation_metrics['accuracy']:.4f} (Δ={test_ablation_metrics['accuracy'] - test_metrics['accuracy']:+.4f})")
            logger.info(f"    F1 Macro:  {test_metrics['f1_macro']:.4f} → {test_ablation_metrics['f1_macro']:.4f} (Δ={test_ablation_metrics['f1_macro'] - test_metrics['f1_macro']:+.4f})")
            logger.info(f"    ROC-AUC:   {test_metrics['roc_auc']:.4f} → {test_ablation_metrics['roc_auc']:.4f} (Δ={test_ablation_metrics['roc_auc'] - test_metrics['roc_auc']:+.4f})")

            # Simpan metrik ablation per fold
            fold_ablation_result = {
                "fold": fold,
                "held_out_group": group_name,
                "test_ablation_loss": test_ablation_metrics["loss"],
                "test_ablation_accuracy": test_ablation_metrics["accuracy"],
                "test_ablation_roc_auc": test_ablation_metrics["roc_auc"],
                "test_ablation_f1_macro": test_ablation_metrics["f1_macro"],
                "test_ablation_precision_macro": test_ablation_metrics["precision_macro"],
                "test_ablation_recall_macro": test_ablation_metrics["recall_macro"],
            }
            all_fold_ablation_test_metrics.append(fold_ablation_result)

            # Simpan classification report dan confusion matrix ablation per fold
            all_fold_ablation_classification_reports.append({
                "fold": fold,
                "held_out_group": group_name,
                "report": ablation_report_dict,
            })

        # Simpan metrik test per fold
        fold_test_result = {
            "fold": fold,
            "held_out_group": group_name,
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_f1_macro": test_metrics["f1_macro"],
            "test_precision_macro": test_metrics["precision_macro"],
            "test_recall_macro": test_metrics["recall_macro"],
        }
        all_fold_test_metrics.append(fold_test_result)

        # Simpan classification report per fold
        all_fold_classification_reports.append({
            "fold": fold,
            "held_out_group": group_name,
            "report": report_dict,
        })

        # =================================================================
        # 3e. Cleanup memori GPU
        # =================================================================
        if df_test_ablation is not None and test_ablation_loader is not None:
            del df_test_ablation, test_ablation_loader
        del model, best_model, train_loader, val_loader, test_loader
        del df_train, df_val, df_train_val
        accelerator.free_memory()
        gc.collect()
        if accelerator.device.type == "cuda":
            torch.cuda.empty_cache()

        # Hapus model sementara
        if accelerator.is_main_process:
            shutil.rmtree(fold_tmp_dir, ignore_errors=True)
        accelerator.wait_for_everyone()

    # =========================================================================
    # 4. Agregasi hasil CV
    # =========================================================================
    logger.info("=" * 60)
    logger.info("RINGKASAN GSS-CV")
    logger.info("=" * 60)

    per_fold_df = pd.DataFrame(all_fold_test_metrics)

    # Kolom metrik untuk dihitung mean ± std
    metric_cols = [
        "test_loss", "test_accuracy", "test_roc_auc",
        "test_f1_macro", "test_precision_macro", "test_recall_macro"
    ]

    summary = {}
    for col in metric_cols:
        mean_val = per_fold_df[col].mean()
        std_val = per_fold_df[col].std()
        summary[f"{col}_mean"] = mean_val
        summary[f"{col}_std"] = std_val
        logger.info(f"  {col}: {mean_val:.4f} ± {std_val:.4f}")

    # Agregasi hasil ablation CV (hanya jika ada)
    ablation_summary = None
    if len(all_fold_ablation_test_metrics) > 0:
        logger.info("\n" + "-" * 40)
        logger.info("RINGKASAN ABLATION GSS-CV")
        logger.info("-" * 40)

        ablation_per_fold_df = pd.DataFrame(all_fold_ablation_test_metrics)
        ablation_metric_cols = [
            "test_ablation_loss", "test_ablation_accuracy", "test_ablation_roc_auc",
            "test_ablation_f1_macro", "test_ablation_precision_macro", "test_ablation_recall_macro"
        ]

        ablation_summary = {}
        for col in ablation_metric_cols:
            mean_val = ablation_per_fold_df[col].mean()
            std_val = ablation_per_fold_df[col].std()
            ablation_summary[f"{col}_mean"] = mean_val
            ablation_summary[f"{col}_std"] = std_val
            logger.info(f"  {col}: {mean_val:.4f} ± {std_val:.4f}")

        # Perbandingan ringkasan normal vs ablation
        logger.info("\n" + "-" * 40)
        logger.info("PERBANDINGAN NORMAL vs ABLATION (Mean)")
        logger.info("-" * 40)
        logger.info(f"  Accuracy:  {summary['test_accuracy_mean']:.4f} → {ablation_summary['test_ablation_accuracy_mean']:.4f} (Δ={ablation_summary['test_ablation_accuracy_mean'] - summary['test_accuracy_mean']:+.4f})")
        logger.info(f"  F1 Macro:  {summary['test_f1_macro_mean']:.4f} → {ablation_summary['test_ablation_f1_macro_mean']:.4f} (Δ={ablation_summary['test_ablation_f1_macro_mean'] - summary['test_f1_macro_mean']:+.4f})")
        logger.info(f"  ROC-AUC:   {summary['test_roc_auc_mean']:.4f} → {ablation_summary['test_ablation_roc_auc_mean']:.4f} (Δ={ablation_summary['test_ablation_roc_auc_mean'] - summary['test_roc_auc_mean']:+.4f})")

    # =========================================================================
    # 5. Simpan Hasil ke CSV (hanya di main process)
    # =========================================================================
    if accelerator.is_main_process:
        # Simpan hasil test per fold
        per_fold_path = os.path.join(cv_result_path, "cv_per_fold_results.csv")
        per_fold_df.to_csv(per_fold_path, index=False)
        logger.info(f"\nHasil per-fold disimpan ke: {per_fold_path}")

        # Simpan ringkasan CV (mean ± std)
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(cv_result_path, "cv_summary_results.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Ringkasan CV disimpan ke: {summary_path}")

        # Simpan hasil training per epoch per fold
        train_df = pd.DataFrame(all_fold_train_metrics)
        train_path = os.path.join(cv_result_path, "cv_train_results.csv")
        train_df.to_csv(train_path, index=False)
        logger.info(f"Hasil training per epoch disimpan ke: {train_path}")

        # Simpan classification report per fold
        for item in all_fold_classification_reports:
            report_df = pd.DataFrame(item["report"]).transpose()
            report_filename = f"classification_report_fold_{item['fold']}.csv"
            report_path = os.path.join(cv_result_path, report_filename)
            report_df.to_csv(report_path, index=True)

        logger.info(f"Classification reports disimpan ke: {cv_result_path}")

        # Simpan hasil ablation CV (hanya jika ada)
        if len(all_fold_ablation_test_metrics) > 0:
            # Simpan hasil ablation test per fold
            ablation_per_fold_df = pd.DataFrame(all_fold_ablation_test_metrics)
            ablation_per_fold_path = os.path.join(cv_result_path, "cv_ablation_per_fold_results.csv")
            ablation_per_fold_df.to_csv(ablation_per_fold_path, index=False)
            logger.info(f"\nHasil ablation per-fold disimpan ke: {ablation_per_fold_path}")

            # Simpan ringkasan ablation CV (mean ± std)
            if ablation_summary is not None:
                ablation_summary_df = pd.DataFrame([ablation_summary])
                ablation_summary_path = os.path.join(cv_result_path, "cv_ablation_summary_results.csv")
                ablation_summary_df.to_csv(ablation_summary_path, index=False)
                logger.info(f"Ringkasan ablation CV disimpan ke: {ablation_summary_path}")

            # Simpan classification report ablation per fold
            for item in all_fold_ablation_classification_reports:
                report_df = pd.DataFrame(item["report"]).transpose()
                report_filename = f"ablation_classification_report_fold_{item['fold']}.csv"
                report_path = os.path.join(cv_result_path, report_filename)
                report_df.to_csv(report_path, index=True)

            logger.info(f"Ablation classification reports disimpan ke: {cv_result_path}")

    logger.info("\n🎉 GSS-CV selesai!")
