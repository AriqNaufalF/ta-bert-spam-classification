import os
from dotenv import load_dotenv
from typing import Callable, overload, Any

# Load environment variables from .env file
load_dotenv()

MISSING = object()  # Sentinel value for missing environment variables

@overload
def get_env(key: str, *, default: str) -> str: ...

@overload
def get_env(key: str) -> str: ...

@overload
def get_env[T](key: str, var_type: Callable[[str], T], *, default: T) -> T: ...

@overload
def get_env[T](key: str, var_type: Callable[[str], T]) -> T: ...

def get_env(key: str, var_type: Any = str, *, default: Any = MISSING) -> Any:
    """
    Get environment variable and raise exception if not found.
    
    Args:
        key: Environment variable name
        var_type: Type to convert the value to (str, int, float)
        default: Default value to return if the environment variable is not set (optional)
    Raises:
        ValueError: If environment variable is not set
    """
    value = os.getenv(key)
    if value is None:
        if default is not MISSING:
            return default
        raise ValueError(f"Environment variable '{key}' is required but not set")
    
    try:
        return var_type(value)
    except ValueError:
        raise ValueError(f"Environment variable '{key}' has invalid value '{value}' for type {var_type.__name__}")
    

class Config:
    """
    Kelas untuk menyimpan semua konfigurasi hyperparameter dan path file.
    Semua nilai harus tersedia di file .env.
    """

    # --- Path Dataset ---
    INDONESIAN_CSV = get_env('INDONESIAN_CSV')
    ENGLISH_CSV = get_env('ENGLISH_CSV')

    # --- Model ---
    MODEL_NAME = get_env('MODEL_NAME')

    # --- Tokenizer ---
    # Panjang maksimum token input.
    MAX_LENGTH = get_env('MAX_LENGTH', int)

    # --- Hyperparameter Training (Optimized with Optuna) ---
    BATCH_SIZE = get_env('BATCH_SIZE', int)
    LEARNING_RATE = get_env('LEARNING_RATE', float)
    WEIGHT_DECAY = get_env('WEIGHT_DECAY', float)
    DROPOUT_RATE = get_env('DROPOUT_RATE', float)
    EPOCHS = get_env('EPOCHS', int)
    WARMUP_RATIO = get_env('WARMUP_RATIO', float)
    PATIENCE = get_env('PATIENCE', int)

    # --- Data Split ---
    RANDOM_SEED = get_env('RANDOM_SEED', int)

    # --- Output ---
    # Direktori untuk menyimpan model yang sudah di-fine-tune
    OUTPUT_DIR = get_env('OUTPUT_DIR')


    # Hyperparameter tuning ranges (untuk Optuna) dari .env
    LR_RANGE = (get_env('LR_MIN', float), get_env('LR_MAX', float))
    WEIGHT_DECAY_RANGE = (get_env('WEIGHT_DECAY_MIN', float), get_env('WEIGHT_DECAY_MAX', float))
    DROPOUT_RATE_RANGE = (get_env('DROPOUT_RATE_MIN', float), get_env('DROPOUT_RATE_MAX', float))
    WARMUP_RATIO_RANGE = (get_env('WARMUP_RATIO_MIN', float), get_env('WARMUP_RATIO_MAX', float))