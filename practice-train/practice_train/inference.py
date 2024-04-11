import os
import logging
import json
import numpy
import mlflow
from azure.ai.ml import MLClient
from azure.identity import EnvironmentCredential

logging.info("Logging into Azure workspace")
ml_client = MLClient(
        credential=EnvironmentCredential(),
        subscription_id=os.getenv("subscription_id"),
        resource_group_name=os.getenv("resource_group"),
        workspace_name=os.getenv("workspace_name"),
    )

def init():
    global model
    logging.info("Init start")

    # Path to the saved PyTorch model within the MLflow run artifacts
    # Pick the latest version of the model
    latest_model_version = max(
        [int(m.version) for m in ml_client.models.list(name=os.getenv("MODEL_NAME"))]
    )
    model_uri = f"models:/{os.getenv("MODEL_NAME")}/{latest_model_version}"

    # Load the PyTorch model using MLflow
    model = mlflow.pytorch.load_model(model_uri)

    # Set the model to evaluation mode
    model.eval()

    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("model 1: request received")
    data = json.loads(raw_data)["data"]
    data = numpy.array(data)
    result = model.predict(data)
    logging.info("Request processed")
    return result.tolist()
