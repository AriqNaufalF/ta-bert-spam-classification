import logging, sys

def setup_logger():
  logger = logging.getLogger('my_logger')
  logger.setLevel(logging.INFO)
  logger.propagate = False

  # Cek apakah handler sudah ada agar tidak duplikat
  if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

  return logger

logger = setup_logger()