import logging
import os

log_format = "%(asctime)s - %(levelname)s - %(message)s"
log_file_path = os.path.join("logs", "app.log")  # Create a 'logs/' directory if not exists

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
log_file_path = os.path.join(parent_dir, "logs", "app.log")

# Ensure the logs directory exists
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path)
    ]
)

logger = logging.getLogger(__name__)