import logging
import os
import random

import numpy as np
import torch
import wandb


def SeedEverything(seed=808):
    """Method to seed everything."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_config(file_path):
    """Open and read yaml config.

    Args:
        file_path (str): path to config file

    Returns:
        dict: config file
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def setup_logging(say_my_name="debug"):
    """Set up logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        handlers=[
            logging.StreamHandler(),  # Output log messages to console
            logging.FileHandler(
                f"logs/{say_my_name}.log"
            ),  # Save log messages to a file
        ],
    )


def set_wandb(config):
    # Set up wandb logging
    if config["debug"] == 1:
        project_name = "debug_llm"
    else:
        project_name = "kaggle_llm"

    wandb.init(
        # Set the project where this run will be logged
        project=project_name,
        # Set up the run name
        name=(
            f"{config['name']}_lr{config['learning_rate']}_"
            f"epochs{config['epochs']}"
        ),
        # Track hyperparameters and run metadata
        config={
            "learning_rate": config["learning_rate"],
            "epochs": config["epochs"],
        },
    )
