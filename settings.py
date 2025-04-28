import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# logging.basicConfig(level=logging.INFO,
#                     format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)



log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('app.log')  # Log to a file
    ]
)


logger = logging.getLogger(__name__)


