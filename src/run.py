import argparse
from utils import Config
from data import YouTubeSpamDatasetBaseline, YouTubeSpamDataset
from training import train, train_gss_cv, parameter_tuning

def handle_train(args):
    config = Config()
    dataset_class = YouTubeSpamDatasetBaseline if args.baseline else YouTubeSpamDataset
    if args.cv:
        train_gss_cv(config, args.dataset, dataset_class)
    else:
        train(config, args.dataset, dataset_class)

def handle_tune(args):
    config = Config()
    dataset_class = YouTubeSpamDatasetBaseline if args.baseline else YouTubeSpamDataset
    parameter_tuning(config, args.dataset, dataset_class, args.n_trials)


def main():
    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument(
        '--dataset',
        type=str,
        choices=['all', 'id', 'en'],
        default='all',
        help="Pilih dataset yang akan digunakan: 'all' (default), 'id', atau 'en'"
    )
    parent_parser.add_argument(
        '--baseline',
        action='store_true',
        help="Gunakan dataset baseline (hanya komentar) tanpa konteks video"
    )

    parser = argparse.ArgumentParser(description="YouTube Spam Comment Classification dengan BERT")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Subparser for training
    train_parser = subparsers.add_parser("train", parents=[parent_parser], help="Mulai training model BERT untuk klasifikasi spam")
    train_parser.add_argument(
        '--cv',
        action='store_true',
        help="Gunakan Leave-One-Group-Out Cross-Validation (LOGO-CV) untuk evaluasi"
    )
    train_parser.set_defaults(func=handle_train)

    # Subparser for hyperparameter tuning
    tune_parser = subparsers.add_parser("tune", parents=[parent_parser], help="Mulai hyperparameter tuning dengan Optuna")
    tune_parser.add_argument(
        "-n", "--n-trials",
        type=int,
        default=15,
        help="Jumlah trial untuk Optuna (default: 15)"
    )
    tune_parser.set_defaults(func=handle_tune)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()