from utils import Config
from transformers import BertForSequenceClassification
from torch.utils.data import  DataLoader
from transformers import BertTokenizer
import pandas as pd
from typing import Type
from data import YouTubeSpamDataset, YouTubeSpamDatasetBaseline

def initialize_model(config: Config) -> BertForSequenceClassification:
    """
    Inisialisasi model BERT dengan bobot pre-trained yang fresh.

    Args:
        config: Objek konfigurasi

    Returns:
        Model BERT yang siap di-fine-tune
    """
    model = BertForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=2,
        hidden_dropout_prob=config.DROPOUT_RATE,
        attention_probs_dropout_prob=config.DROPOUT_RATE,
    )
    return model

def create_dataloader(
    df: pd.DataFrame,
    tokenizer: BertTokenizer,
    config: Config,
    dataset_class: Type[YouTubeSpamDataset | YouTubeSpamDatasetBaseline] = YouTubeSpamDataset,
    shuffle: bool = False,
) -> DataLoader:
    """
    Args:
        df: DataFrame berisi data yang akan dimuat
        tokenizer: BERT tokenizer
        config: Objek konfigurasi
        dataset_class: Kelas Dataset yang akan digunakan (default: YouTubeSpamDataset)
        shuffle: Apakah data perlu diacak (True untuk training)

    Returns:
        DataLoader yang siap digunakan
    """
    dataset = dataset_class(df, tokenizer, config.MAX_LENGTH)
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )