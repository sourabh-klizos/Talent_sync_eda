import logging
import os
from logging.handlers import TimedRotatingFileHandler




parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
log_file_path = os.path.join(parent_dir, "logs", "app.log")


os.makedirs(os.path.dirname(log_file_path), exist_ok=True) # create if not exists


log_format = (
    "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s"
)


logging.basicConfig(
    level=logging.DEBUG,
    format=log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path, mode='a') 
    ],
)

logger = logging.getLogger("app")





