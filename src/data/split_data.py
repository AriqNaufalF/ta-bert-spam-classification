import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from typing import Iterator
from utils import logger

def gss_split(
    df: pd.DataFrame,
    n_splits: int,
    test_size: float,
    group_col: str,
    random_state: int
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Args:
        df: DataFrame berisi data
        n_splits: Jumlah split
        test_size: Ukuran test set (0-1)
        group_col: Nama kolom group
        random_state: Seed untuk random number generator

    Returns:
        Iterator menghasilkan tuple berisi train, dan val set
    """
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    for train_idx, val_idx in gss.split(df, groups=df[group_col]):
        train_set = df.iloc[train_idx].reset_index(drop=True)
        val_set = df.iloc[val_idx].reset_index(drop=True)

        yield train_set, val_set

def log_split_info(
    df: pd.DataFrame,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> None:
    """
    Args:
        df: DataFrame keseluruhan (untuk menghitung persentase)
        df_train: DataFrame training
        df_val: DataFrame validasi
        df_test: DataFrame test
    """
    total = len(df)
    logger.info(f"  Train: {len(df_train)} sampel ({len(df_train)/total*100:.1f}%)")
    logger.info(f"  Val:   {len(df_val)} sampel ({len(df_val)/total*100:.1f}%)")
    logger.info(f"  Test:  {len(df_test)} sampel ({len(df_test)/total*100:.1f}%)")

    # Tampilkan video per split
    train_videos = set(df_train["video_title"].unique())
    val_videos = set(df_val["video_title"].unique())
    test_videos = set(df_test["video_title"].unique())

    logger.info(f"  Train videos ({len(train_videos)}): {sorted(train_videos)}")
    logger.info(f"  Val videos   ({len(val_videos)}): {sorted(val_videos)}")
    logger.info(f"  Test videos  ({len(test_videos)}): {sorted(test_videos)}")

    # Cek data leakage antar split
    leak_tv = train_videos & val_videos
    leak_tt = train_videos & test_videos
    leak_vt = val_videos & test_videos

    if leak_tv or leak_tt or leak_vt:
        logger.warning(f"  ⚠️ PERINGATAN: Data leakage terdeteksi!")
        if leak_tv:
            logger.warning(f"    Video bocor Train↔Val: {leak_tv}")
        if leak_tt:
            logger.warning(f"    Video bocor Train↔Test: {leak_tt}")
        if leak_vt:
            logger.warning(f"    Video bocor Val↔Test: {leak_vt}")
    else:
        logger.info("  ✅ Tidak ada data leakage — video terpisah sempurna antar split!")

    # Distribusi label per split
    for nama, data in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        distribusi = data["label"].value_counts(normalize=True)
        logger.info(
            f"  {nama} — Spam: {distribusi.get(1, 0)*100:.1f}%, "
            f"Ham: {distribusi.get(0, 0)*100:.1f}%"
        )

def logo_split(
    df: pd.DataFrame,
    group_col: str,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, int, str]]:
    """
    Leave-One-Group-Out split: setiap grup unik dijadikan test set sekali.

    Args:
        df: DataFrame berisi data
        group_col: Nama kolom group (misalnya 'video_title')

    Yields:
        (df_train_val, df_test, fold_index, group_name)
        - df_train_val: Data untuk training + validasi
        - df_test: Data test (satu grup)
        - fold_index: Indeks fold (dimulai dari 1)
        - group_name: Nama grup yang dijadikan test set
    """
    logo = LeaveOneGroupOut()
    groups = df[group_col]
    unique_groups = groups.unique()
    n_groups = len(unique_groups)

    logger.info(f"  Leave-One-Group-Out CV: {n_groups} fold (grup: {sorted(unique_groups)})")

    for fold_index, (train_val_idx, test_idx) in enumerate(logo.split(df, groups=groups), start=1):
        df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        group_name = df_test[group_col].iloc[0]

        yield df_train_val, df_test, fold_index, group_name


def log_cv_split_info(
    df: pd.DataFrame,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    fold: int,
    group_name: str,
) -> None:
    """
    Log informasi distribusi split untuk satu fold CV.

    Args:
        df: DataFrame keseluruhan (untuk menghitung persentase)
        df_train: DataFrame training
        df_val: DataFrame validasi
        df_test: DataFrame test (held-out group)
        fold: Nomor fold
        group_name: Nama grup yang dijadikan test set
    """
    total = len(df)
    logger.info(f"  --- Fold {fold} | Held-out group: '{group_name}' ---")
    logger.info(f"  Train: {len(df_train)} sampel ({len(df_train)/total*100:.1f}%)")
    logger.info(f"  Val:   {len(df_val)} sampel ({len(df_val)/total*100:.1f}%)")
    logger.info(f"  Test:  {len(df_test)} sampel ({len(df_test)/total*100:.1f}%)")

    # Tampilkan video per split
    train_videos = set(df_train["video_title"].unique())
    val_videos = set(df_val["video_title"].unique())
    test_videos = set(df_test["video_title"].unique())

    logger.info(f"  Train videos ({len(train_videos)}): {sorted(train_videos)}")
    logger.info(f"  Val videos   ({len(val_videos)}): {sorted(val_videos)}")
    logger.info(f"  Test videos  ({len(test_videos)}): {sorted(test_videos)}")

    # Cek data leakage
    leak_tt = train_videos & test_videos
    leak_vt = val_videos & test_videos

    if leak_tt or leak_vt:
        logger.warning(f"  ⚠️ PERINGATAN: Data leakage terdeteksi di fold {fold}!")
        if leak_tt:
            logger.warning(f"    Video bocor Train↔Test: {leak_tt}")
        if leak_vt:
            logger.warning(f"    Video bocor Val↔Test: {leak_vt}")
    else:
        logger.info("  ✅ Tidak ada data leakage — video terpisah sempurna antar split!")

    # Distribusi label per split
    for nama, data in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        distribusi = data["label"].value_counts(normalize=True)
        logger.info(
            f"  {nama} — Spam: {distribusi.get(1, 0)*100:.1f}%, "
            f"Ham: {distribusi.get(0, 0)*100:.1f}%"
        )
