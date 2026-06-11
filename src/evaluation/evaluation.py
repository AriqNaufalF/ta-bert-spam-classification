import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from accelerate import Accelerator
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
from tqdm.autonotebook import tqdm

@torch.no_grad()
def evaluate(
        model: BertForSequenceClassification,
        dataloader: DataLoader,
        accelerator: Accelerator,
        class_weights: torch.Tensor | None = None,
        deskripsi: str = "Evaluasi",
    ) -> dict:
    """
    Args:
        model: Model BERT yang akan dievaluasi
        dataloader: DataLoader untuk data evaluasi (validation/test)
        accelerator: Accelerator instance untuk multi-GPU support
        class_weights: Bobot per kelas untuk loss function
        deskripsi: Label untuk progress bar (misalnya "Validasi" atau "Test")

    Returns:
        Dictionary berisi metrik evaluasi:
        - loss: Rata-rata loss
        - f1_macro: F1-score macro (metrik utama)
        - f1_weighted: F1-score weighted
        - precision_macro: Precision macro
        - recall_macro: Recall macro
        - confusion_matrix: Confusion matrix sebagai numpy array
        - all_preds: Semua prediksi
        - all_labels: Semua label ground truth
    """
    # Set model ke mode evaluasi (nonaktifkan dropout)
    model.eval()
    total_loss = 0.0
    jumlah_batch = 0

    # List untuk menyimpan semua prediksi, label dan probabilitas
    all_preds = []
    all_labels = []
    all_probs = []
    
    if class_weights is not None:
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(accelerator.device)
        )
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    progress_bar = tqdm(
        dataloader, 
        desc=f"[{deskripsi}]", 
        leave=True,
        disable=not accelerator.is_local_main_process,
    )

    for batch in progress_bar:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["label"]

        # Forward pass (tanpa gradient computation karena @torch.no_grad)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = outputs.logits

        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        jumlah_batch += 1

        # Ambil prediksi: kelas dengan probabilitas tertinggi
        preds = torch.argmax(logits, dim=1)

        # Ambil probabilitas untuk kelas spam (1) — untuk hitung ROC-AUC
        probs = torch.softmax(logits, dim=1)[:, 1]

        # Ambil prediksi dan label dari semua proses (multi-GPU)
        # gather_for_metrics() menghapus padding duplikat yang ditambahkan Accelerate
        gathered_preds, gathered_labels, gathered_probs = accelerator.gather_for_metrics(
            (preds, labels, probs)
        )

        # Simpan ke list (pindahkan ke CPU)
        all_preds.extend(gathered_preds.cpu().numpy())
        all_labels.extend(gathered_labels.cpu().numpy())
        all_probs.extend(gathered_probs.cpu().numpy())

    # Hitung metrik evaluasi
    avg_loss = total_loss / jumlah_batch

    # Akurasi — persentase total prediksi yang benar
    accuracy = accuracy_score(all_labels, all_preds)

    # ROC-AUC — kemampuan model membedakan antar kelas (menggunakan probabilitas)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # Jika hanya ada satu kelas (jarang terjadi), ROC-AUC tidak bisa dihitung
        roc_auc = 0.0


    # F1-Score Macro — metrik utama, memberikan bobot sama ke setiap kelas
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    # F1-Score Weighted — mempertimbangkan jumlah sampel per kelas
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    # Precision — seberapa akurat prediksi "spam"
    prec_macro = precision_score(all_labels, all_preds, average="macro")

    # Recall — seberapa banyak spam yang berhasil terdeteksi
    rec_macro = recall_score(all_labels, all_preds, average="macro")

    # Confusion Matrix — melihat distribusi True/False Positive/Negative
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "confusion_matrix": cm,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "all_probs": all_probs,
    }
