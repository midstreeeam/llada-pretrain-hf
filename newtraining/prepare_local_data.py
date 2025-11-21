import argparse
import torch
import random
import os
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

class PrepareLocalData:
    def __init__(self, 
                 input_file: str,
                 output_dir: str,
                 tokenizer_name: str = "GSAI-ML/LLaDA-8B-Instruct",
                 max_seq_length: int = 4096,
                 id_mask_token: int = 126336,
                 chunks_per_file: int = 10000):
        
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.tokenizer_name = tokenizer_name
        self.max_seq_length = max_seq_length
        self.id_mask_token = id_mask_token
        self.chunks_per_file = chunks_per_file
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.init_tokenizer()

    def init_tokenizer(self):
        print(f"Loading tokenizer from {self.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def process_chunk(self, chunk_tokens, device, eps=1e-3):
        chunk_tensor = torch.tensor(chunk_tokens, dtype=torch.int32, device=device)
        t = random.random()
        p_mask = (1.0 - eps) * t + eps
        mask = torch.rand(chunk_tensor.size(0), device=device) < p_mask
        noisy = chunk_tensor.clone()
        noisy[mask] = self.id_mask_token
        return {
            "t": t,
            "input_ids": chunk_tensor,
            "noisy_input_ids": noisy,
            "mask": mask
        }

    def prepare(self):
        print(f"==== Loading Data from {self.input_file} ====")
        # Load local jsonl file
        dataset = load_dataset("json", data_files=self.input_file, split="train")
        
        print("==== Tokenizing and Processing ====")
        device = torch.device("cpu")
        
        processed = []
        current_chunk = []
        chunk_count = 0
        file_count = 0
        
        # Process in a streaming-like fashion to avoid OOM on large datasets
        # We iterate and tokenize on the fly or in batches if we used map, 
        # but for simplicity and memory safety with large files, simple iteration is fine.
        # To speed up, we can use map with batched=True for tokenization.
        
        def tokenize_function(examples):
            return self.tokenizer(examples["text"])

        # Tokenize efficiently
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )

        for example in tqdm(tokenized_dataset, desc="Creating Chunks"):
            tokens = example["input_ids"]
            current_chunk.extend(tokens)

            while len(current_chunk) >= self.max_seq_length:
                # Random length strategy from original script
                if random.random() < 0.01:
                    L = random.randint(1, self.max_seq_length)
                else:
                    L = self.max_seq_length
                
                chunk_tokens = current_chunk[:L]
                processed.append(self.process_chunk(chunk_tokens, device))
                current_chunk = current_chunk[L:]
                chunk_count += 1

                if chunk_count % self.chunks_per_file == 0:
                    self.save_batch(processed, file_count)
                    processed = []
                    file_count += 1
        
        # Process remaining tokens in current_chunk
        if current_chunk:
            processed.append(self.process_chunk(current_chunk, device))
            chunk_count += 1

        # Save remaining
        if processed:
            self.save_batch(processed, file_count)
            
        print(f"==== Finished. Total chunks: {chunk_count} ====")

    def save_batch(self, data, file_idx):
        file_path = self.output_dir / f"processed_chunk_{file_idx:06d}.pt"
        torch.save(data, file_path)
        print(f"✅ Saved {len(data)} chunks to {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to input .jsonl file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .pt files")
    parser.add_argument("--tokenizer", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    
    args = parser.parse_args()
    
    preparer = PrepareLocalData(
        input_file=args.input_file,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer
    )
    preparer.prepare()
