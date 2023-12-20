import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("/files/my_trains/mamba_FPB/complete")
tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template

model = MambaLMHeadModel.from_pretrained("/files/my_trains/mamba_FPB/complete/", device="cuda", dtype=torch.float16)

messages = []
while True:
    user_message = input("\nYour message: ")

    text = f"""Classify the setiment of the following news headlines as either `positive`, `neutral`, or `negative`.\n
    Headline: {user_message}\n
    Classification:"""

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")

    out = model.generate(
        input_ids=input_ids, 
        max_length=2000, 
        temperature=0.9, 
        top_p=0.7, 
        eos_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    print("Classification:", decoded.split('Classification: ')[-1].strip())