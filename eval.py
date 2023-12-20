import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import argparse
from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset
import evaluate
from tqdm import tqdm


accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load('recall')
f1 = evaluate.load("f1")
    

def label_to_string(label):
    if label == 1:
        return 'neutral'
    elif label == 2:
        return 'positive'
    else:
        return 'negative'


def label_to_int(label):
    if label.lower() == 'neutral':
        return 1 
    elif label == 'positive':
        return 2
    elif label.lower() == 'negative':
        return 3
            
def run(args):
    # dataset = load_from_disk(f'./dataset')
    dataset = load_dataset('winddude/finacial_pharsebank_66agree_split')
    dataset = dataset['test']
    
    model = MambaLMHeadModel.from_pretrained(args.model, dtype=torch.bfloat16, device="cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    
    predictions = []
    references = []
    
    for x in tqdm(dataset):
        text = f"""Classify the setiment of the following news headlines as either `positive`, `neutral`, or `negative`.\n
        Headline: {x['sentence']}\n
        Classification:"""

        input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")

        out = model.generate(
            input_ids=input_ids, 
            max_length=250, 
            temperature=0.9, 
            top_p=0.7, 
            eos_token_id=tokenizer.eos_token_id
        )

        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        extracted = decoded.split('Classification: ')[-1].strip()
        references.append(x['label'])
        predictions.append(label_to_int(extracted))

    # calc metrics   
    results = {
        'accuracy': accuracy.compute(predictions=predictions, references=references),
        'precision': precision.compute(predictions=predictions, references=references, average='micro'),
        'recall': recall.compute(predictions=predictions, references=references, average='micro'),
        'f1': f1.compute(predictions=predictions, references=references, average='micro')
    }
    
    print(results)
    breakpoint()
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, default="/files/my_trains/mamba_FPB/complete")
    parser.add_argument("--tokenizer", type=str, default="/files/my_trains/mamba_FPB/complete")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="/files/my_trains/mamba_FPB")
    
    args = parser.parse_args()
    
    run(args)