# How to run Llama-2 on CPU after fine-tuning with LoRA

Running Large Language Models (LLMs) on the edge is a fascinating area of research, and opens up many use cases. With libraries like [ggml](https://github.com/ggerganov/ggml) coming on to the scene, it is now possible to get models anywhere from 1 billion to 13 billion parameters to run locally on a laptop with relatively low latency.

In this tutorial, we are going to walk step by step how to fine tune Llama-2 with LoRA, export it to ggml, and run it real time on CPU.

We assume you know the benefits of fine-tuning, have a basic understanding of Llama and LoRA, and are excited about running models at the edge 😎. Let's get started.

*Note: All of these library are being updated and changing at the speed of light, so this formula worked for me in October 2023, and the library versions will be documented in the `requirements.txt`*

All the code for this tutorial can be found at [https://github.com/Oxen-AI/Llama-Fine-Tune](https://github.com/Oxen-AI/Llama-Fine-Tune).

## Running Llama-2 on CPU

Before we get into fine-tuning, let's start by seeing how easy it is to run Llama-2 on GPU with LangChain and it's `CTransformers` interface.

To get up and running quickly [TheBloke](https://huggingface.co/TheBloke) has done a lot of work exporting these models to GGML format for us. You can find many variations of Llama-2 in GGML format for [download here](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main).


```bash
pip install langchain
pip install ctransformers
```

```python
from langchain.llms import CTransformers
from langchain.callbacks.base import BaseCallbackHandler

# Handler that prints each new token as it is computed
class NewTokenHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"{token}", end="", flush=True)

# Local CTransformers wrapper for Llama-2-7B-Chat
llm = CTransformers(
    model="models/llama-2-7b-chat.ggmlv3.q8_0.bin", # Location of downloaded GGML model
    model_type="llama", # Model type Llama
    stream=True,
    callbacks=[NewTokenHandler()],
    config={'max_new_tokens': 256, 'temperature': 0.01}
)

# Accept user input
while True:
    prompt = input('> ')
    output = llm(prompt)
    print(output)
```

The first time you run inference, it will take a second to load the model into memory, but after that you can see the tokens being printed in real time.

## Dad-Joke LLM

For this example, we are going to see if we can fine-tune Llama-2 to complete witty jokes. Comedy is a fun example of fine-tuning as everyone has a different sense of humor. What may be funny to me, might not be funny to you. In theory there are many ways to fine tune an LLM to have your exact sense of humor, and be less dry and sterile than a language model that was simply trained to predict the next word.

## The Prompt

We will be working with a sample prompt to get us a jumping off point. When evaluating LLMs, I like to have a use case and a few specific examples in mind, and do a quick sanity check of how the base model performs.

It is hard to quantitatively evaluate LLMs in the context of comedy without manually labeling many generated outputs, but we will get to that later.

In this case, we will try to get Llama-2 to be a "hilarious, dry, one-liner standup comedian"

```
You are a hilarious, dry, one-liner standup comedian. Please complete the following joke setup with a punchline: Why was the headmaster worried?
```

When we first ask the llama-2-7b-chat-q8_0 model, the LLM has a very generic response.

```
Because he had a lot of _______________
```

It kind of got the gist, but couldn't complete the task. Not too funny if you ask me.

## N-Shot Prompt

The next logical step is to create an N-Shot prompt, and prime the model with some more examples. This is going to be way quicker to iterate on than an entire fine-tuning process, and is worth a shot.

## Fine-Tuning with LoRA



```python
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_int8_training,
)
from trl import SFTTrainer
from transformers import LlamaForCausalLM, LlamaTokenizer
import sys

if len(sys.argv) != 3:
    print("Usage: python fine_tune.py <dataset.jsonl> <results_dir>")
    exit()

dataset_file = sys.argv[1]
output_dir = sys.argv[2]

dataset = load_dataset("parquet", data_files={'train': dataset_file})
# dataset = load_dataset("json", data_files={'train': dataset_file})
dataset = dataset['train']
print(dataset)

base_model_name = "meta-llama/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

device_map = {"": 0}

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
    use_auth_token=True
)
base_model.config.use_cache = False

# More info: https://github.com/huggingface/transformers/pull/24906
base_model.config.pretraining_tp = 1 

# peft_config = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=64,
#     bias="none",
#     task_type="CAUSAL_LM",
# )

def create_peft_config(model):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_int8_training,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules = ["q_proj", "v_proj"]
    )

    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model, lora_config = create_peft_config(base_model)


training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    # gradient_accumulation_steps=16,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    # learning_rate=1.41e-5,
    learning_rate=2e-4,
    logging_steps=1,
    # max_steps=10000
)

max_seq_length = 1024

trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()

import os
output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)
```