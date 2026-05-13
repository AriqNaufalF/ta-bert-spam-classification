import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    BertForSequenceClassification,
)
from accelerate import Accelerator
from tqdm.autonotebook import tqdm

def train_one_epoch(
        model: BertForSequenceClassification,
        dataloader: DataLoader,
        optimizer: AdamW,
        scheduler,
        accelerator: Accelerator,
        class_weights: torch.Tensor,
        epoch: int,
    ) -> float:
    """
    Args:
        model: Model BERT yang sedang di-training
        dataloader: DataLoader untuk data training (sudah di-prepare oleh Accelerate)
        optimizer: Optimizer AdamW (sudah di-prepare oleh Accelerate)
        scheduler: Learning rate scheduler dengan warmup (sudah di-prepare)
        accelerator: Accelerator instance untuk multi-GPU support
        class_weights: Bobot per kelas untuk loss function
        epoch: Nomor epoch saat ini (untuk tampilan progress bar)

    Returns:
        Rata-rata loss selama satu epoch
    """
    # Set model ke mode training (aktifkan dropout dan batch normalization)
    model.train()
    total_loss = 0.0
    jumlah_batch = 0

    # Loss function dengan class weights untuk handling class imbalance
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=class_weights.to(accelerator.device)
    )

    # Progress bar hanya ditampilkan di proses utama (menghindari duplikasi di multi-GPU)
    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch} [Training]",
        leave=True,
        disable=not accelerator.is_local_main_process,
    )

    for batch in progress_bar:
        # Accelerate sudah menangani device placement untuk batch dari prepared dataloader
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["label"]

        # Reset gradien dari iterasi sebelumnya
        optimizer.zero_grad()

        # Forward pass — dapatkan logits dari model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = outputs.logits

        # Hitung loss menggunakan CrossEntropyLoss dengan class weights
        loss = loss_fn(logits, labels)

        # Backward pass via Accelerate — menangani gradient scaling untuk multi-GPU
        accelerator.backward(loss)

        # Gradient clipping — batasi gradien agar tidak terlalu besar (mencegah exploding gradient)
        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update parameter model
        optimizer.step()

        # Update learning rate sesuai schedule
        scheduler.step()

        # Akumulasi loss untuk menghitung rata-rata
        total_loss += loss.item()
        jumlah_batch += 1

        # Tampilkan loss saat ini di progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Hitung rata-rata loss per epoch
    avg_loss = total_loss / jumlah_batch
    return avg_loss
