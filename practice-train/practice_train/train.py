import argparse
from dotenv import load_dotenv
import os

import logging
import mlflow
import pandas as pd
import transformers
from azure.ai.ml import MLClient
from azure.identity import EnvironmentCredential
from transformers import EarlyStoppingCallback, IntervalStrategy

from initialise_model import initialise

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id",
    type=str,
    help="Path to the training data",
    default="openai-community/gpt2",
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # Get arguments
    logging.info("Loading arguments and env vars")
    load_dotenv()
    args = parser.parse_args()

    logging.info("Logging into Azure workspace")
    ml_client = MLClient(
        credential=EnvironmentCredential(),
        subscription_id=os.getenv("subscription_id"),
        resource_group_name=os.getenv("resource_group"),
        workspace_name=os.getenv("workspace_name"),
    )

    logging.info("Tracking experiment with mlflow")
    mlflow_tracking_uri = ml_client.workspaces.get(
        ml_client.workspace_name
    ).mlflow_tracking_uri

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Loading model and tokenizer
    model, tokenizer = initialise(args.model_id)

    logging.info("Loading data")
    train_data, eval_data = (
        pd.read_pickle("./data/train")["prompt"].apply(tokenizer).to_list(),
        pd.read_pickle("./data/eval")["prompt"].apply(tokenizer).to_list(),
    )

    # Define training args
    training_args = transformers.TrainingArguments(
        auto_find_batch_size=True,
        num_train_epochs=4,
        learning_rate=2e-4,
        save_total_limit=4,
        logging_steps=10,
        output_dir="./output",
        evaluation_strategy="steps",
        save_strategy=IntervalStrategy.STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Start training logging with MLflow
    logging.info("Start mlflow run")
    mlflow.set_experiment("shakespearean-gpt")
    mlflow.autolog()
    with mlflow.start_run() as run:

        # Add inference script
        mlflow.log_artifact("./code/score.py")

        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            args=training_args,
            data_collator=transformers.DataCollatorForLanguageModeling(
                tokenizer, mlm=False
            ),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        model.config.use_cache = False

        logging.info("Training...")
        trainer.train()

        # Push merged model to model registry
        logged_model_info = mlflow.pytorch.log_model(
            pytorch_model=trainer.model,
            artifact_path="shakespearean-model",
            input_example=train_data,
            registered_model_name="tiny-shakespeare",
        )

        logging.info("Registering model...")
        mlflow.register_model(
            f"runs:/{run.info.run_id}/shakespearean-model", "tiny-shakespeare"
        )

        mlflow.end_run()
        logging.info("End run")

        # Use predefined question-answering metrics to evaluate our model.
        results = mlflow.evaluate(
            logged_model_info.model_uri,
            eval_data,
            targets="ground_truth",
            model_type="question-answering",
        )

        # Evaluation result for each data record is available in `results.tables`.
        eval_table = results.tables["eval_results_table"]
