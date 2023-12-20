import torch
import argparse

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, TrainingArguments
from trainer.mamba_trainer import MambaTrainer
from datasets import load_dataset


def run(args):
        
    model = MambaLMHeadModel.from_pretrained(args.model, dtype=torch.bfloat16, device="cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 200

    def label_to_string(label):
        if label == 1:
            return 'neutral'
        elif label == 2:
            return 'positive'
        else:
            return 'negative'
        
    def tokenization(example):
        text = f"""Classify the setiment of the following news headlines as either `positive`, `neutral`, or `negative`.\n
        Headline: {example['sentence']}\n
        Classification: {label_to_string(example['label'])}\n
        """
        return tokenizer(text)

    dataset = load_dataset('financial_phrasebank', 'sentences_66agree', split="train")
    dataset = dataset.train_test_split(test_size=0.1)
    dataset.save_to_disk(f'./dataset')
    dataset = dataset['train']
    dataset = dataset.map(tokenization)
    dataset = dataset.remove_columns(["sentence", "label"])
    
    trainer = MambaTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=TrainingArguments(
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim=args.optim,
            output_dir=args.output_dir,
            logging_steps=50,
            save_steps=500,
            save_strategy='steps'
        )
    )

    trainer.train(
        #resume_from_checkpoint=True
    )
        
    trainer.save_model(output_dir=f'{args.output_dir}/complete')


if __name__ == "__main__":
       
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="state-spaces/mamba-2.8b")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="/files/my_trains/mamba_FPB")
    args = parser.parse_args()

    run(args)