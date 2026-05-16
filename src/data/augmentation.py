import re
import nltk
import pandas as pd
import nlpaug.augmenter.word as naw
from utils import logger
from typing import Literal

# Token khusus yang tidak boleh diganti sinonim
_SPECIAL_TOKENS = ["[URL]", "[EMAIL]", "[TIMESTAMP]", "[USER]", "[HASHTAG]"]

# Mapping bahasa ke kode lang WordNet (OMW-1.4)
_LANG_MAP = {
    "en": "eng",
    "id": "ind",
}

# Cache augmenter per bahasa agar tidak dibuat ulang setiap pemanggilan
_augmenter_cache: dict[str, naw.SynonymAug] = {}


def _ensure_nltk_resource(resource_path: str, download_name: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name, quiet=True, raise_on_error=True)


_ensure_nltk_resource("corpora/wordnet", "wordnet")
_ensure_nltk_resource("corpora/omw-1.4", "omw-1.4")
_ensure_nltk_resource(
    "taggers/averaged_perceptron_tagger",
    "averaged_perceptron_tagger",
)
_ensure_nltk_resource(
    "taggers/averaged_perceptron_tagger_eng",
    "averaged_perceptron_tagger_eng",
)

def _get_augmenter(lang: Literal["en", "id"], replacement_ratio: float) -> naw.SynonymAug | None:
    """
    Mengembalikan instance SynonymAug untuk bahasa yang diberikan.
    Instance di-cache agar tidak dibuat ulang untuk setiap baris.

    Args:
        lang: Kode bahasa ('en' atau 'id').
        replacement_ratio: Proporsi kata yang diganti (aug_p).

    Returns:
        Instance naw.SynonymAug, atau None jika bahasa tidak didukung.
    """
    lang_code = _LANG_MAP.get(lang)
    if lang_code is None:
        return None

    cache_key = f"{lang_code}_{replacement_ratio}"
    if cache_key not in _augmenter_cache:
        _augmenter_cache[cache_key] = naw.SynonymAug(
            aug_src="wordnet",
            lang=lang_code,
            aug_p=replacement_ratio,
            stopwords=_SPECIAL_TOKENS,
        )

    return _augmenter_cache[cache_key]


def synonym_replacement(text: str, lang: Literal["en", "id"], replacement_ratio: float = 0.2) -> str:
    """
    Melakukan augmentasi teks dengan mengganti sejumlah kata dengan sinonimnya
    secara acak menggunakan nlpaug SynonymAug (WordNet).

    Token khusus seperti [URL], [EMAIL], [TIMESTAMP], [USER], [HASHTAG]
    tidak akan diganti (didaftarkan sebagai stopwords pada augmenter).

    Args:
        text: Teks input yang akan diaugmentasi.
        lang: Bahasa teks ('en' untuk Inggris, 'id' untuk Indonesia).
        replacement_ratio: Proporsi kata yang akan diganti (default 0.2 = 20%).

    Returns:
        Teks hasil augmentasi. Mengembalikan teks asli jika augmentasi gagal
        atau bahasa tidak didukung.
    """
    if not text or not text.strip():
        return text

    augmenter = _get_augmenter(lang, replacement_ratio)
    if augmenter is None:
        return text

    try:
        result = augmenter.augment(text)
        # nlpaug.augment() mengembalikan list[str]; ambil elemen pertama
        if isinstance(result, list):
            result = result[0] if result else text
        else:
            result = str(result)
        # nlpaug kadang menambahkan spasi di dalam token khusus: [ URL ] -> [URL]
        result = re.sub(r'\[\s*(\w+)\s*\]', r'[\1]', result)
        return result
    except Exception:
        # Jika augmentasi gagal (misal teks terlalu pendek), kembalikan teks asli
        return text


def _log_augmentation_examples(
    df_original: pd.DataFrame,
    df_augmented: pd.DataFrame,
    n: int = 1,
) -> None:
    """
    Menampilkan contoh hasil augmentasi per kategori:
    - ID spam  (language='id', label=1)
    - EN ham   (language='en', label=0)
    - EN spam  (language='en', label=1)

    Args:
        df_original:  DataFrame target sebelum augmentasi (indeks sama).
        df_augmented: DataFrame target setelah augmentasi (indeks sama).
        n:            Jumlah contoh per kategori (default 1).
    """
    categories = [
        ("ID Spam",  (df_original["language"] == "id") & (df_original["label"] == 1)),
        ("EN Ham",   (df_original["language"] == "en") & (df_original["label"] == 0)),
        ("EN Spam",  (df_original["language"] == "en") & (df_original["label"] == 1)),
    ]

    logger.info("-" * 60)
    logger.info("  [CONTOH HASIL AUGMENTASI]")

    for category_name, mask in categories:
        indices = df_original.index[mask].tolist()
        if not indices:
            logger.info(f"\n  [{category_name}] — tidak ada data untuk kategori ini")
            continue

        samples = indices[:n]
        logger.info(f"\n  [{category_name}]")
        for i, idx in enumerate(samples, start=1):
            original_text = df_original.at[idx, "comment"]
            augmented_text = df_augmented.at[idx, "comment"]
            logger.info(f"    Contoh #{i}:")
            logger.info(f"      Original  : {original_text}")
            logger.info(f"      Augmented : {augmented_text}")


def augment_train_data(
    df_train: pd.DataFrame,
    replacement_ratio: float = 0.2,
    random_state: int = 42,
    log_examples: bool = False,
) -> pd.DataFrame:
    """
    Melakukan augmentasi data training dengan Random Replacement Synonym
    menggunakan nlpaug SynonymAug (WordNet / OMW-1.4) pada:
    - Data dengan language = 'id' dan label = 1 (spam Indonesia)
    - Data dengan language = 'en' (semua label Inggris)

    Baris yang diaugmentasi ditambahkan sebagai baris baru (duplikat yang
    telah dimodifikasi), sehingga data asli tetap utuh.

    Args:
        df_train: DataFrame data training sebelum augmentasi.
        replacement_ratio: Proporsi kata yang diganti per teks (default 0.2).
        random_state: Seed untuk reproducibilitas (default 42).

    Returns:
        DataFrame hasil augmentasi (data asli + baris augmented).
    """
    import random
    random.seed(random_state)

    # =========================================================================
    # Log distribusi SEBELUM augmentasi
    # =========================================================================
    logger.info("=" * 60)
    logger.info("DATA AUGMENTATION — Random Replacement Synonym")
    logger.info("=" * 60)
    logger.info(f"  Rasio penggantian per teks : {replacement_ratio:.0%}")
    logger.info(f"  Target augmentasi          :")
    logger.info(f"    1. language='id' & label=1 (spam Bahasa Indonesia)")
    logger.info(f"    2. language='en' (semua label Bahasa Inggris)")
    logger.info("-" * 60)
    logger.info("  [SEBELUM AUGMENTASI]")
    logger.info(f"  Total baris                : {len(df_train)}")
    logger.info(
        "  Distribusi label per bahasa:\n"
        + df_train.groupby(["language", "label"])
        .size()
        .rename("count")
        .to_string()
    )

    # =========================================================================
    # Seleksi baris target augmentasi
    # =========================================================================
    mask_id_spam = (df_train["language"] == "id") & (df_train["label"] == 1)
    mask_en = df_train["language"] == "en"
    mask_target = mask_id_spam | mask_en

    df_target = df_train[mask_target].copy()
    logger.info(f"\n  Jumlah baris yang diaugmentasi: {len(df_target)}")

    # =========================================================================
    # Terapkan augmentasi pada kolom 'comment' (kolom teks utama)
    # =========================================================================
    def _augment_row(row):
        lang = row["language"]
        row = row.copy()
        row["comment"] = synonym_replacement(
            row["comment"], lang=lang, replacement_ratio=replacement_ratio
        )
        return row

    df_augmented = pd.DataFrame(df_target.apply(_augment_row, axis=1))

    # =========================================================================
    # Log contoh hasil augmentasi (3 kategori: ID spam, EN ham, EN spam)
    # =========================================================================
    if log_examples:
        _log_augmentation_examples(df_target, df_augmented, 5)

    # =========================================================================
    # Gabungkan data asli dengan data augmentasi
    # =========================================================================
    df_result = pd.concat([df_train, df_augmented], ignore_index=True)

    # =========================================================================
    # Log distribusi SETELAH augmentasi
    # =========================================================================
    logger.info("-" * 60)
    logger.info("  [SETELAH AUGMENTASI]")
    logger.info(f"  Total baris                : {len(df_result)}")
    logger.info(
        f"  Baris baru (augmented)     : {len(df_result) - len(df_train)}"
    )
    logger.info(
        "  Distribusi label per bahasa:\n"
        + df_result.groupby(["language", "label"])
        .size()
        .rename("count")
        .to_string()
    )
    logger.info("=" * 60)

    return df_result
