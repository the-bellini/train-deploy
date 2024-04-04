# Train-Deploy

Train-Deploy is a repo to practice training and deploying an LLM on Azure virtual machines

The aim is to keep the Ops code-based rather than use the out-of-the-box Azure ML services.

The training pipeline:
  1. Start compute, ssh into the virtual machine (vm) and then clone the repo OR start compute in Azure notebooks and use the terminal to clone the repo 
  3. Add .env variables from the .env.example file
  4. Run "sudo snap install docker" to install docker on the vm
  5. Run "sudo docker-compose up --build" to containerise the training script and begin training
  7. Training is logged via MLFlow and model artefacts are stored in Model Registry
