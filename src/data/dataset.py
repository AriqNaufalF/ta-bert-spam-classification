from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import torch

# Custom Dataset untuk Pytorch

# Custom PyTorch Dataset untuk data komentar YouTube.

# Dataset ini menangani tokenisasi BERT dengan format input:

# `[CLS] comment [SEP] judul: deskripsi [SEP]`

# Strategi tokenisasi:
# - Segmen A (token_type_id=0): comment → teks yang diklasifikasikan
# - Segmen B (token_type_id=1): video_title + video_description → konteks video
# - Prioritas truncation: deskripsi dipotong duluan, lalu komentar, kemudian judul

# Alasan menggunakan custom tokenisasi (bukan tokenizer bawaan):
# - BERT tokenizer standar hanya mendukung 2 segmen (text, text_pair)
# - Dibutuhkan 3 segmen (judul, deskripsi, komentar) dengan prioritas
#   truncation khusus

# Proses tokenisasi manual:
# 1. Tokenisasi masing-masing bagian komentar dan konteks secara terpisah
# 2. Hitung token yang tersedia setelah dikurangi token spesial ([CLS], [SEP])
# 3. Alokasikan token sesuai prioritas:
#    - Komentar: diambil utuh
#    - Judul: diambil utuh jika memungkinkan
#    - Deskripsi: dipotong dari belakang jika token tidak cukup
# 4. Gabungkan semua bagian dengan token spesial yang sesuai
# 5. Buat attention mask dan token type IDs
class YouTubeSpamDataset(Dataset):
    """
    Args:
        dataframe: DataFrame berisi kolom comment, video_title, video_description, label
        tokenizer: BERT tokenizer dari HuggingFace
        max_length: Panjang maksimum token (default: 256)
    """

    def __init__(
        self, dataframe: pd.DataFrame, tokenizer: BertTokenizer, max_length: int = 256
    ):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Mengembalikan jumlah sampel dalam dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Mengambil satu sampel dan melakukan tokenisasi.

        Args:
            idx: Indeks sampel dalam dataset

        Returns:
            Dictionary berisi input_ids, attention_mask, token_type_ids, dan label
        """
        row = self.data.iloc[idx]

        # Ambil teks dari masing-masing kolom
        comment = str(row["comment"])
        title = str(row["video_title"])
        description = str(row["video_description"])
        label = int(row["label"])

        # =====================================================================
        # Tokenisasi masing-masing bagian secara terpisah (tanpa token spesial)
        # =====================================================================
        # truncation=True agar tokenizer memotong teks yang melebihi model_max_length (256)
        # secara otomatis tanpa menampilkan warning. Hasil tokenisasi ini kemudian
        # akan dipotong lagi oleh logika alokasi token di bawah untuk menyesuaikan
        # budget gabungan [CLS] comment [SEP] judul: deskripsi [SEP] = 256 token.
        context_tokens = self.tokenizer.encode(
            f"{title}: {description}", add_special_tokens=False, truncation=True, max_length=self.max_length
        )
        comment_tokens = self.tokenizer.encode(
            comment, add_special_tokens=False, truncation=True, max_length=self.max_length
        )

        # =====================================================================
        # Hitung alokasi token
        # Format: [CLS] comment [SEP] judul: deskripsi [SEP]
        # Token spesial: 1 ([CLS]) + 2 ([SEP]) = 3 token
        # =====================================================================
        special_tokens = 3
        remaining_token = self.max_length - special_tokens

        # Prioritas alokasi: Komentar (utuh)> Judul (sisa) > Deskripsi (sisa)
        # Langkah 1: Alokasikan token untuk komentar
        actual_comment_tokens = comment_tokens[:remaining_token]
        remaining_token -= len(actual_comment_tokens)

        # Langkah 2: Alokasikan sisa token untuk konteks (judul dan deskripsi)
        actual_context_tokens = context_tokens[:remaining_token]
        # =====================================================================
        # Susun input_ids dengan format:
        # [[CLS] comment [SEP] judul: deskripsi [SEP]
        # =====================================================================
        cls_id = self.tokenizer.cls_token_id  # Token [CLS] = 101
        sep_id = self.tokenizer.sep_token_id  # Token [SEP] = 102

        # Prioritaskan token untuk komentar
        input_ids = [cls_id] + actual_comment_tokens + [sep_id]

        # Tambahkan segmen konteks jika masih ada token tersisa
        # Jika tidak ada sisa input hanya berupa [CLS] comment [SEP]
        if len(actual_context_tokens) > 0:
            input_ids += actual_context_tokens + [sep_id]

        # =====================================================================
        # Buat token_type_ids untuk membedakan segmen
        # =====================================================================
        # Segmen A: dari [CLS] + komentar + [SEP]
        panjang_segmen_a = len(actual_comment_tokens) + 2
        # Segmen B: konteks + [SEP] terakhir
        panjang_segmen_b = len(actual_context_tokens) + (1 if len(actual_context_tokens) > 0 else 0)

        token_type_ids = ([0] * panjang_segmen_a) + ([1] * panjang_segmen_b)

        # =====================================================================
        # Buat attention_mask (1 = token nyata, 0 = padding)
        # =====================================================================
        attention_mask = [1] * len(input_ids)

        # =====================================================================
        # Padding — tambahkan token [PAD] hingga mencapai max_length
        # Token [PAD] memiliki id=0 pada BERT tokenizer
        # =====================================================================
        panjang_padding = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * panjang_padding
        attention_mask += [0] * panjang_padding
        token_type_ids += [0] * panjang_padding

        # Konversi ke tensor PyTorch
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


# Custom Dataset untuk Pytorch

# Custom PyTorch Dataset untuk data komentar YouTube.

# Dataset ini menangani tokenisasi BERT dengan format input:

# `[CLS] comment [SEP]`

# Proses tokenisasi:
# 1. Tokenisasi bagian komentar saja tanpa token spesial
# 2. Hitung token yang tersedia setelah dikurangi token spesial ([CLS], [SEP])
# 3. Alokasikan token sesuai prioritas:
#    - Komentar: diambil utuh
# 4. Gabungkan semua bagian dengan token spesial yang sesuai
# 5. Buat attention mask dan token type IDs
class YouTubeSpamDatasetBaseline(Dataset):
    """
    Args:
        dataframe: DataFrame berisi kolom comment, video_title, video_description, label
        tokenizer: BERT tokenizer dari HuggingFace
        max_length: Panjang maksimum token (default: 256)
    """

    def __init__(
        self, dataframe: pd.DataFrame, tokenizer: BertTokenizer, max_length: int = 256
    ):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Mengembalikan jumlah sampel dalam dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Mengambil satu sampel dan melakukan tokenisasi.

        Args:
            idx: Indeks sampel dalam dataset

        Returns:
            Dictionary berisi input_ids, attention_mask, token_type_ids, dan label
        """
        row = self.data.iloc[idx]

        # Ambil teks dari masing-masing kolom
        comment = str(row["comment"])
        label = int(row["label"])

        # =====================================================================
        # Tokenisasi tanpa token spesial
        # =====================================================================
        # truncation=True agar tokenizer memotong teks yang melebihi model_max_length (256)
        # secara otomatis tanpa menampilkan warning. Hasil tokenisasi ini kemudian
        # akan dipotong lagi oleh logika alokasi token di bawah untuk menyesuaikan
        # budget gabungan [CLS] comment [SEP] = 256 token.

        comment_tokens = self.tokenizer.encode(
            comment, add_special_tokens=False, truncation=True, max_length=self.max_length
        )

        # =====================================================================
        # Hitung alokasi token
        # Format: [CLS] comment [SEP]
        # Token spesial: 1 ([CLS]) + 1 ([SEP]) = 2 token
        # =====================================================================
        special_tokens = 2
        remaining_token = self.max_length - special_tokens

        # Prioritas alokasi: Komentar (utuh)
        actual_comment_tokens = comment_tokens[:remaining_token]
        remaining_token -= len(actual_comment_tokens)

        # =====================================================================
        # Susun input_ids dengan format:
        # [[CLS] comment [SEP]
        # =====================================================================
        cls_id = self.tokenizer.cls_token_id  # Token [CLS] = 101
        sep_id = self.tokenizer.sep_token_id  # Token [SEP] = 102

        # Prioritaskan token untuk komentar
        input_ids = [cls_id] + actual_comment_tokens + [sep_id]

        # =====================================================================
        # Buat token_type_ids untuk membedakan segmen
        # =====================================================================
        # Segmen A: dari [CLS] + komentar + [SEP]

        token_type_ids = [0] * len(input_ids)   

        # =====================================================================
        # Buat attention_mask (1 = token nyata, 0 = padding)
        # =====================================================================
        attention_mask = [1] * len(input_ids)

        # =====================================================================
        # Padding — tambahkan token [PAD] hingga mencapai max_length
        # Token [PAD] memiliki id=0 pada BERT tokenizer
        # =====================================================================
        panjang_padding = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * panjang_padding
        attention_mask += [0] * panjang_padding
        token_type_ids += [0] * panjang_padding

        # Konversi ke tensor PyTorch
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }
