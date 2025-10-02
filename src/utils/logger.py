import logging
import os

def get_logger(name: str = "app", log_file: str = "logs/app.log") -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Avoid adding multiple handlers if get_logger called many times
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger
