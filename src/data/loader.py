from utils import Config, logger
import pandas as pd
from data import preprocess_text 

COLUMNS_NEEDED = [
    "author",
    "comment",
    "published_at",
    "video_description",
    "video_title",
    "label",
    "language",
]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan pembersihan data
    Args:
        df (pd.DataFrame): dataframe yang akan dibersihkan

    Returns:
        pd.DataFrame: dataframe yang sudah dibersihkan
    """
    df_clean = df[COLUMNS_NEEDED].copy()
    
    # Hapus duplikat exact (semua kolom sama)
    jumlah_sebelum = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    jumlah_dihapus = jumlah_sebelum - len(df_clean)
    logger.info(f"  Duplikat exact dihapus: {jumlah_dihapus} baris")
    logger.info(f"  Total data setelah deduplikasi: {len(df_clean)} baris")

    df_clean["comment"] = df_clean["comment"].apply(preprocess_text)
    df_clean["video_title"] = df_clean["video_title"].apply(preprocess_text)
    df_clean["video_description"] = df_clean["video_description"].apply(preprocess_text)

    # Pastikan tidak ada nilai null pada kolom kritis
    # Isi nilai kosong dengan string kosong agar tidak error saat tokenisasi
    for col in ["comment", "video_title", "video_description"]:
        df_clean[col] = df_clean[col].fillna("")

    return df_clean

def load_id_dataset(config: Config) -> pd.DataFrame:
    """
    Memuat dataset Indonesia dari file CSV.

    Args:
        config (Config): Objek konfigurasi yang berisi path file CSV

    Returns:
        pd.DataFrame: DataFrame yang sudah dimuat dan siap untuk diproses lebih lanjut
    """
    logger.info("Memuat dataset Indonesia...")
    df_id = pd.read_csv(config.INDONESIAN_CSV)
    # Tambahkan label bahasa 'id' untuk dataset Indonesia
    df_id["language"] = "id"
    logger.info(f"  Dataset Indonesia dimuat: {len(df_id)} baris")


    df_id_cleaned = clean_data(df_id)

    return df_id_cleaned

def load_en_dataset(config: Config) -> pd.DataFrame:
    """
    Memuat dataset Inggris dari file CSV.

    Args:
        config (Config): Objek konfigurasi yang berisi path file CSV

    Returns:
        pd.DataFrame: DataFrame yang sudah dimuat dan siap untuk diproses lebih lanjut
    """
    logger.info("Memuat dataset Inggris...")
    df_en = pd.read_csv(config.ENGLISH_CSV)
    # Tambahkan label bahasa 'en' untuk dataset Inggris
    df_en["language"] = "en"
    logger.info(f"  Dataset Inggris dimuat: {len(df_en)} baris")

    # Standarisasi nama kolom dataset Inggris agar sama dengan dataset Indonesia
    df_en = df_en.rename(
        columns={
            "AUTHOR": "author",
            "CONTENT": "comment",
            "DATE": "published_at",
            "Video Title": "video_title",
            "Video Description": "video_description",
            "CLASS": "label",
        }
    )

    df_en_cleaned = clean_data(df_en)

    return df_en_cleaned

# Fungsi Pemuatan dan Persiapan Data

# Langkah-langkah:
#   1. Muat CSV Indonesia dan Inggris
#   2. Standarisasi nama kolom (kolom Inggris di-rename agar seragam)
#   3. Tambahkan kolom 'language' untuk identifikasi bahasa
#   4. Gabungkan kedua dataset menjadi satu DataFrame
#   5. Hapus duplikat exact (seluruh kolom sama)
#   6. Terapkan preprocessing teks pada kolom comment, video_title, video_description
def load_all_datasets(config: Config) -> pd.DataFrame:
    """
    Memuat kedua file CSV dataset dan menyiapkannya untuk training.

    Args:
        config: Objek konfigurasi yang berisi path file CSV

    Returns:
        DataFrame gabungan yang sudah bersih dan siap untuk splitting
    """
    df_id = load_id_dataset(config)
    df_en = load_en_dataset(config)

    # Gabungkan kedua dataset menjadi satu DataFrame
    df = pd.concat([df_id, df_en], ignore_index=True)
    logger.info(f"Total data setelah digabungkan: {len(df)} baris")

    return df
