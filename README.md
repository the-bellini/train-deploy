# Train-Deploy

Train-Deploy is a repo to practice training and deploying an LLM on Azure virtual machines

The aim is to keep the Ops code-based rather than use the out-of-the-box Azure ML services.

The training pipeline:
  1. Start compute
  2. ssh into virtual machine and clone repo
  3. Dockerise the training script and run the training script
  4. Log training via MLFlow
  5. Store model artefacts in Model Registry
