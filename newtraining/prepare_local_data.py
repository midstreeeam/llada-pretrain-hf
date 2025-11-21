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
        # chunk_tokens is already a tensor or list of ints
        if not torch.is_tensor(chunk_tokens):
            chunk_tensor = torch.tensor(chunk_tokens, dtype=torch.int32, device=device)
        else:
            chunk_tensor = chunk_tokens.to(device=device, dtype=torch.int32)
            
        t = random.random()
        p_mask = (1.0 - eps) * t + eps
        mask = torch.rand(chunk_tensor.size(0), device=device) < p_mask
        noisy = chunk_tensor.clone()
        noisy[mask] = self.id_mask_token
        return {
            "t": t,
            "input_ids": chunk_tensor.cpu(), # Move back to CPU to save RAM/VRAM when accumulating
            "noisy_input_ids": noisy.cpu(),
            "mask": mask.cpu()
        }

    def prepare(self, batch_size: int = 1000):
        print(f"==== Loading Data from {self.input_file} ====")
        import json
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"==== Tokenizing and Processing on {device} ====")
        
        processed = []
        current_chunk = []
        chunk_count = 0
        file_count = 0
        
        # Count lines for tqdm
        total_lines = 0
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for _ in f:
                total_lines += 1
                
        print(f"Total lines: {total_lines}")

        batch_texts = []
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=total_lines, desc="Processing")
            
            for line in f:
                try:
                    record = json.loads(line)
                    text = record.get("text", "")
                    if text:
                        batch_texts.append(text)
                except json.JSONDecodeError:
                    pass
                
                pbar.update(1)
                
                if len(batch_texts) >= batch_size:
                    # Tokenize batch
                    encodings = self.tokenizer(batch_texts, add_special_tokens=True)
                    for tokens in encodings["input_ids"]:
                        current_chunk.extend(tokens)
                        
                        while len(current_chunk) >= self.max_seq_length:
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
                    
                    batch_texts = []
            
            pbar.close()

        # Process remaining batch
        if batch_texts:
            encodings = self.tokenizer(batch_texts, add_special_tokens=True)
            for tokens in encodings["input_ids"]:
                current_chunk.extend(tokens)
                while len(current_chunk) >= self.max_seq_length:
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
