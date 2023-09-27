import argparse
import gc
import logging
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTrainedTokenizerBase,
)

from dataset_utils import DataCollatorForMultipleChoice, preprocess
from metrics import compute_metrics, validation_MAP
from utils import SeedEverything, read_config, set_wandb, setup_logging


def main(args):
    """Main script

    Args:
        args (argparse.Namespace): arguments to run the script.
    """
    # Access the values of the arguments
    config_file = args.config
    mode = args.mode

    # Read config file
    config = read_config(config_file)

    # Set up logging messages
    setup_logging(config["name"])
    logging.info("Started the program.")

    # Enable garbage collector and seed everything
    gc.enable()
    SeedEverything(config["seed"])

    # Run the train part
    if mode == "train":
        set_wandb(config)

        # Read the data to pandas
        logging.info("Loading the data.")
        df_train = pd.read_csv(config["data_path"])
        df_train = df_train.drop(columns="source")
        if config["debug"]:
            df_train = df_train.fillna("").sample(100)

        df_valid = pd.read_csv(config["valid_path"])

        # Create datasets
        logging.info("Starting the tokenization.")
        tokenizer = AutoTokenizer.from_pretrained(config["MODEL"])
        dataset_valid = Dataset.from_pandas(df_valid)
        dataset = Dataset.from_pandas(df_train)
        dataset = dataset.remove_columns(["__index_level_0__"])

        tokenized_dataset_valid = dataset_valid.map(
            preprocess,
            remove_columns=[
                "prompt",
                "context",
                "A",
                "B",
                "C",
                "D",
                "E",
                "answer",
            ],
        )
        tokenized_dataset = dataset.map(
            preprocess,
            remove_columns=[
                "prompt",
                "context",
                "A",
                "B",
                "C",
                "D",
                "E",
                "answer",
            ],
        )

        # Create the model
        logging.info("Loading the model.")
        model = AutoModelForMultipleChoice.from_pretrained(config["MODEL"])

        training_args = TrainingArguments(
            warmup_ratio=config["warmup_ratio"],
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=1,
            per_device_eval_batch_size=2,
            num_train_epochs=config["epochs"],
            report_to="none",
            output_dir=config["output_dir"],
            overwrite_output_dir=True,
            fp16=True,
            logging_steps=config["logging_steps"],
            evaluation_strategy="epoch",
            save_strategy=config["save_strategy"],
            load_best_model_at_end=False,
            metric_for_best_model="map@3",
            lr_scheduler_type=config["lr_scheduler_type"],
            weight_decay=config["weight_decay"],
            save_total_limit=config["save_total_limit"],
            report_to="wandb",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_dataset_valid,
            compute_metrics=compute_metrics,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )

        logging.info("Training is started.")
        trainer.train()
        trainer.save_model(config["name"])
        logging.info("Finished training, saving the model.")

    elif mode == "validate":
        # Load the model checkpoint
        logging.info("Loading validation checkpoint.")
        model = AutoModelForMultipleChoice.from_pretrained(
            config["model_checkpoint_path"]
        )
        trainer = Trainer(model=model)
        logging.info("Model is loaded.")

        # Load the data
        logging.info("Loading the data.")
        test_df = pd.read_csv(config["valid_path"])
        tokenized_test_dataset = Dataset.from_pandas(test_df).map(
            preprocess,
            remove_columns=["prompt", "context", "A", "B", "C", "D", "E"],
        )
        logging.info("Loaded the data.")

        logging.info("Starting predicting.")
        test_predictions = trainer.predict(tokenized_test_dataset).predictions
        predictions_as_ids = np.argsort(-test_predictions, 1)
        predictions_as_answer_letters = np.array(list("ABCDE"))[
            predictions_as_ids
        ]
        test_df["prediction"] = [
            " ".join(row) for row in predictions_as_answer_letters[:, :3]
        ]

        score = validation_MAP(test_df.prediction.values, test_df.answer.values)
        logging.info("Validation MAP@3: %s", score)

    logging.info("Finished the program.")


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Training script with YAML config."
    )
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "validate"],
        default="train",
        help="Mode: train or validate",
    )
    # Parse the command-line arguments
    args = parser.parse_args()
    # Run main script with arguments
    main(args)
