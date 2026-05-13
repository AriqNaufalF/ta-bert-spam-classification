# Klasifikasi Spam Komentar YouTube dengan BERT

Proyek ini bertujuan untuk mengklasifikasikan komentar spam pada platform YouTube menggunakan model BERT multilingual. Proyek ini mendukung proses pelatihan dan pencarian parameter (*hyperparameter tuning*) menggunakan Optuna. Model dapat dilatih dengan menyertakan konteks video (judul, deskripsi) atau murni hanya menggunakan teks komentar sebagai *baseline*.

## Persiapan dan Instalasi

1. **Instalasi Kebutuhan**
   Pastikan sudah menginstal Python di sistem. Instal semua *package* yang dibutuhkan berdasarkan file `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

2. **Pengaturan Konfigurasi (.env)**
   Proyek ini menggunakan *environment variables* untuk memuat konfigurasi. Gunakan file `.env.example` sebagai templat.
   - Salin file `.env.example` dan simpan dengan nama `.env`.
   - Pada Windows (Command Prompt), gunakan:
     ```cmd
     copy .env.example .env
     ```
   - Pada Linux / Mac / Git Bash:
     ```bash
     cp .env.example .env
     ```
   - Buka file `.env`, kemudian sesuaikan nilai variabel konfigurasi sesuai dengan kebutuhan.

3. **Unduh Dataset**
   Unduh dataset dari [kaggle](https://www.kaggle.com/datasets/ariqnf/indonesian-english-spam-comment-dataset). Simpan file dataset (format `.csv`) ke dalam folder `./data` atau lokasi yang ditentukan di `.env`.

## Cara Menjalankan Proyek

Jalankan file `src/run.py` untuk memulai beberapa proses yang tersedia. Skrip ini menyediakan beberapa perintah utama beserta opsi tambahannya. Jika environment mendukung multi-GPU maka gunakan tools `accelerate launch` untuk menjalankan proses. Gunakan opsi `--num_processes` untuk menentukan jumlah GPU yang akan digunakan.

> **Catatan**: Direkomendasikan untuk menjalankan program pada perangkat/environment yang mendukung akselerasi GPU (misalnya menggunakan Google Colab atau Kaggle Notebook).

### 1. Training Model (`train`)

Perintah ini digunakan untuk memulai proses pelatihan model (*training*)

```bash
# Menggunakan python biasa
python src/run.py train

# Menggunakan accelerate
accelerate launch --num_processes=2 src/run.py train 
```

**Opsi:**
- `--baseline`: Jika opsi ini ditambahkan, model hanya akan dilatih menggunakan teks komentar saja tanpa menyertakan konteks video.

**Contoh Penggunaan:**
```bash
# Melatih model dengan menyertakan konteks video
python src/run.py train

accelerate launch --num_processes=2 src/run.py train

# Melatih model HANYA dengan komentar (tanpa konteks video)
python src/run.py train --baseline

accelerate launch --num_processes=2 src/run.py train --baseline
```

### 2. Hyperparameter Tuning (`tune`)

Perintah ini digunakan untuk memulai proses pencarian nilai parameter terbaik (*hyperparameter tuning*) menggunakan library Optuna.

**Sintaks Dasar:**
```bash
# Menggunakan python biasa
python src/run.py tune

# Menggunakan accelerate
accelerate launch --num_processes=2 src/run.py tune
```

**Opsi yang tersedia:**
- `-n`, `--n-trials` `<INT>`: Menentukan jumlah percobaan (*trials*) yang akan dijalankan oleh Optuna. Nilai *default* adalah `15`.
- `--baseline`: Mirip dengan opsi pada bagian `train`, jika *flag* ini digunakan, *tuning* akan dilakukan menggunakan dataset *baseline* (tanpa menyertakan informasi konteks video).

**Contoh Penggunaan:**
```bash
# Melakukan tuning sebanyak 15 kali (default)
python src/run.py tune

accelerate launch --num_processes=2 src/run.py tune

# Melakukan tuning sebanyak 30 kali dengan dataset lengkap
python src/run.py tune -n 30

accelerate launch --num_processes=2 src/run.py tune -n 30

# Melakukan tuning sebanyak 20 kali menggunakan dataset baseline
python src/run.py tune --n-trials 20 --baseline

accelerate launch --num_processes=2 src/run.py tune --n-trials 20 --baseline
```

## Menu Bantuan 

Untuk melihat rincian perintah secara langsung di terminal, tambahkan opsi `--help` atau `-h`:

```bash
# Bantuan umum mengenai skrip
python src/run.py -h

# Bantuan spesifik untuk argumen 'train'
python src/run.py train --help

# Bantuan spesifik untuk argumen 'tune'
python src/run.py tune --help
```
