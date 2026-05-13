
import gc, logging, optuna
import torch
from transformers import (
    BertTokenizer,
    logging as transformers_logging,
)
from typing import Type, Literal
from accelerate import Accelerator
from utils import Config, logger, compute_class_weights
from data import (
    load_all_datasets, 
    load_en_dataset, 
    load_id_dataset, 
    gss_split, 
    YouTubeSpamDataset, 
    YouTubeSpamDatasetBaseline)
from models import initialize_model, create_dataloader
from .fit_model import fit_model


# Pencarian hyperparameter terbaik menggunakan Optuna
def parameter_tuning(
        config: Config, 
        dataset: Literal["all", "id", "en"] = "all",
        bert_dataset_class: Type[YouTubeSpamDataset | YouTubeSpamDatasetBaseline] = YouTubeSpamDataset,
        n_trials: int = 15
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

    # Membagi data menjadi train dan test 80:20
    df_train, df_test = next(
        gss_split(
            df,
            n_splits=1,
            test_size=0.2,
            group_col=GROUP_COL,
            random_state=config.RANDOM_SEED
        )
    )

    # =========================================================================
    # OPTUNA HYPERPARAMETER TUNING
    # =========================================================================
    logger.info("=" * 60)
    logger.info("MEMULAI HYPERPARAMETER TUNING (OPTUNA)")
    logger.info("=" * 60)

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", *config.LR_RANGE, log=True)
        weight_decay = trial.suggest_float("weight_decay", *config.WEIGHT_DECAY_RANGE, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", *config.DROPOUT_RATE_RANGE)
        warmup_ratio = trial.suggest_float("warmup_ratio", *config.WARMUP_RATIO_RANGE)

        config.LEARNING_RATE = lr
        config.WEIGHT_DECAY = weight_decay
        config.DROPOUT_RATE = dropout_rate
        config.WARMUP_RATIO = warmup_ratio

        train_loader = create_dataloader(df_train, tokenizer, config, dataset_class=bert_dataset_class, shuffle=True)
        val_loader = create_dataloader(df_test, tokenizer, config, dataset_class=bert_dataset_class, shuffle=False)

        model_tune = initialize_model(config)
        model_tune.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            vocab_size = len(tokenizer)
            original_vocab_size = vocab_size - len(CUSTOM_SPECIAL_TOKENS)
            model_tune.bert.embeddings.word_embeddings.weight[
                original_vocab_size:
            ].normal_(mean=0.0, std=0.02)

        class_weights_tune = compute_class_weights(df_train["label"].to_numpy())

        try:
            _, trial_best_f1 = fit_model(
                model=model_tune,
                train_loader=train_loader,
                val_loader=val_loader,
                tokenizer=tokenizer,
                accelerator=accelerator,
                config=config,
                class_weights=class_weights_tune,
                output_dir="",
                save_model=False,
                trial=trial,
            )
            return trial_best_f1
        finally:
            del model_tune, train_loader, val_loader
            accelerator.free_memory()
            gc.collect()
            if accelerator.device.type == "cuda":
                torch.cuda.empty_cache()

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    if not accelerator.is_local_main_process:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(objective, n_trials=n_trials)

    best_lr = study.best_params["learning_rate"]
    best_wd = study.best_params["weight_decay"]
    best_do = study.best_params["dropout_rate"]
    best_wr = study.best_params["warmup_ratio"]

    logger.info("=" * 60)
    logger.info(f"TUNING SELESAI. Best Params:")
    logger.info(f"  LR: {best_lr:.2e}, Weight Decay: {best_wd:.4f}")
    logger.info(f"  Dropout Rate: {best_do:.2f}, Warmup Ratio: {best_wr:.2f}")
    logger.info("=" * 60)

    config.LEARNING_RATE = best_lr
    config.WEIGHT_DECAY = best_wd
    config.DROPOUT_RATE = best_do
    config.WARMUP_RATIO = best_wr