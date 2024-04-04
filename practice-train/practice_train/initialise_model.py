# import torch
import logging
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer  # , BitsAndBytesConfig
from accelerate import disk_offload


def initialise(model_id: str):
    """
    Initializes and returns a language model and its tokenizer configured with BitsAndBytes and LoRA adaptations.

    This function performs the following steps:
    1. Configures BitsAndBytes to improve model loading and execution efficiency. This includes setting 4-bit loading
       options, quantization types, and compute data types.
    2. Loads the model specified by `model_id` with the defined BitsAndBytes configuration. The model is set to auto-map
       to the available device and allows execution of remote code.
    3. Loads the corresponding tokenizer for the model and sets its pad token to the end-of-sequence token.
    4. Prepares the model for k-bit training.
    5. Configures and applies LoRA (Low-Rank Adaptation) settings to the model. This includes setting LoRA parameters
       like rank (r), alpha value, target modules, dropout rate, bias settings, and specifying the task type.

    Parameters:
    model_id (str): The identifier of the pre-trained language model to be loaded.

    Returns:
    tuple: A tuple containing two elements:
        - The first element is the LoRA-configured language model.
        - The second element is the corresponding tokenizer.
    """
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     load_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        trust_remote_code=True,
        # quantization_config=bnb_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # logging.info("Prepare model for 4 bit training")
    # model = prepare_model_for_kbit_training(model)

    # config = LoraConfig(
    #     r=8,
    #     # lora_alpha=32,
    #     target_modules=[
    #         "q_proj",
    #         "o_proj",
    #         "k_proj",
    #         "v_proj",
    #         "gate_proj",
    #         "up_proj",
    #         "down_proj",
    #     ],
    #     # lora_dropout=0.05,
    #     # bias="none",
    #     task_type="CAUSAL_LM",
    # )

    # logging.info(f'Model "{model_id}" and tokenizer loaded')
    # return get_peft_model(model, config), tokenizer
    return model, tokenizer
