import logging
import os
import sys
import pandas as pd
from basic_logger import BasicLogger
import time
import json

if sys.path[-1] != os.getcwd():
    sys.path.append(os.getcwd())


def create_dir(dir_list):
    assert isinstance(dir_list, list)
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)


class TrainLogger:
    def __init__(self, config):
        """
       Initializes the logger and creates a structured save directory based on the configuration.
       """
        # --- Core Fix: Safely get the correct keys from the configuration ---
        self.model_type = config.get('model_type', 'unknown_model')
        self.dataset_name = config.get('dataset_name', 'unknown_dataset')
        # New: Get our custom experiment tag, defaults to an empty string if not present
        self.tag = config.get('experiment_tag', '')

        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        # --- Construct a clear folder name containing all important information ---
        # e.g., 20250710_103000_full_model_drugbank
        self.train_save_dir = os.path.join(
            config.get('save_dir_base', 'save/default_runs'),
            f"{timestamp_str}_{self.model_type}_{self.dataset_name}"
        )

        # Create all necessary subdirectories
        self.log_dir = os.path.join(self.train_save_dir, 'log')
        self.model_dir = os.path.join(self.train_save_dir, 'model')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # Define save paths for the model and configuration file
        self.model_save_path = os.path.join(self.model_dir, 'best_model.pth')
        self.config_save_path = os.path.join(self.train_save_dir, 'config.json')

        # Save the incoming configuration
        with open(self.config_save_path, 'w') as f:
            json.dump(config, f, indent=4)

        # --- Set up the logger (standard logging module) ---
        log_file_path = os.path.join(self.log_dir, f"{self.model_type}_train.log")

        # Use a unique name for each logger instance to prevent logging conflicts
        self.logger = logging.getLogger(f"{self.model_type}_{timestamp_str}")
        self.logger.setLevel(logging.INFO)

        # Avoid adding duplicate handlers to the same logger
        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            # File Handler
            fh = logging.FileHandler(log_file_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

            # Console Handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        self.performance_metrics = []
        self.info(f"Log save path: {log_file_path}")
        self.info(f"Model save path: {self.model_dir}")

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def record_metrics(self, epoch, metrics_dict):
        metrics_dict['epoch'] = epoch
        self.performance_metrics.append(metrics_dict)
        log_str = f"epoch-{epoch} " + " ".join(
            [f"{k}-{v:.4f}" for k, v in metrics_dict.items() if isinstance(v, (int, float))])
        self.info(log_str)

    def get_performance_metrics(self):
        return self.performance_metrics

    def get_model_path(self):
        return self.model_save_path

    def get_model_dir(self):
        return self.train_save_dir
