import logging
import os
from datetime import datetime

# Generate a unique log file name using the current date and time.
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the directory path for storing logs (current working directory + "logs").
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Create the "logs" directory if it doesn't already exist.
os.makedirs(logs_path, exist_ok=True)

# Complete file path for the log file within the "logs" directory.
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the logging module to log messages to the specified log file.
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Set the log file path.
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Define log message format.
    level=logging.INFO  # Set the logging level to INFO.
)
