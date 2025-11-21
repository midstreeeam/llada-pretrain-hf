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

    def process_batch(self, batch_tokens, device, eps=1e-3):
        # batch_tokens: List[List[int]]
        # We need to pad them to the same length to create a tensor, 
        # but our logic ensures they are mostly max_seq_length. 
        # However, the last chunk might be shorter, or random length strategy might be used.
        # For efficiency, we should only batch chunks of the same length or pad.
        # Given the random length strategy, we might have varying lengths.
        # To vectorize efficiently, we can pad to the max length in the batch.
        
        max_len = max(len(t) for t in batch_tokens)
        batch_size = len(batch_tokens)
        
        # Create tensor [B, L]
        # Initialize with pad_token_id if we had one, but here we can just use 0 or mask token
        # Since we are masking anyway, let's just use 0 for padding and ignore it in loss if needed.
        # But wait, the model expects specific shapes. 
        # The training script handles variable lengths via padding in collator.
        # Here we are saving processed chunks.
        
        # Construct tensor
        input_ids = torch.full((batch_size, max_len), 0, dtype=torch.int32, device=device)
        for i, tokens in enumerate(batch_tokens):
            input_ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.int32, device=device)
            
        # Generate t [B]
        t = torch.rand(batch_size, device=device)
        
        # Generate mask probability p_mask [B]
        p_mask = (1.0 - eps) * t + eps
        
        # Generate mask [B, L]
        # p_mask is [B], we need to broadcast to [B, L]
        rand_vals = torch.rand(input_ids.shape, device=device)
        mask = rand_vals < p_mask.unsqueeze(1)
        
        # Handle padding: ensure padding positions are not masked (optional, but good practice)
        # We need a sequence mask to know real lengths
        seq_lens = torch.tensor([len(t) for t in batch_tokens], device=device)
        # Create range [0, max_len)
        range_tensor = torch.arange(max_len, device=device).unsqueeze(0) # [1, L]
        # valid_mask [B, L]
        valid_mask = range_tensor < seq_lens.unsqueeze(1)
        
        # Only mask valid positions
        mask = mask & valid_mask
        
        # Create noisy input
        noisy_input_ids = input_ids.clone()
        noisy_input_ids[mask] = self.id_mask_token
        
        # Move back to CPU and convert to list of dicts
        input_ids_cpu = input_ids.cpu()
        noisy_input_ids_cpu = noisy_input_ids.cpu()
        mask_cpu = mask.cpu()
        t_cpu = t.cpu()
        
        results = []
        for i in range(batch_size):
            l = len(batch_tokens[i])
            results.append({
                "t": t_cpu[i].item(),
                "input_ids": input_ids_cpu[i, :l],
                "noisy_input_ids": noisy_input_ids_cpu[i, :l],
                "mask": mask_cpu[i, :l]
            })
            
        return results

    def prepare(self, batch_size: int = 1000, gpu_batch_size: int = 1024):
        print(f"==== Loading Data from {self.input_file} ====")
        import json
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"==== Tokenizing and Processing on {device} ====")
        
        processed = []
        current_chunk = []
        chunk_count = 0
        file_count = 0
        
        pending_chunks = [] # List of List[int]
        
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
                            pending_chunks.append(chunk_tokens)
                            current_chunk = current_chunk[L:]
                            
                            if len(pending_chunks) >= gpu_batch_size:
                                batch_results = self.process_batch(pending_chunks, device)
                                processed.extend(batch_results)
                                chunk_count += len(batch_results)
                                pending_chunks = []
                                
                                # Save if needed
                                while len(processed) >= self.chunks_per_file:
                                    to_save = processed[:self.chunks_per_file]
                                    processed = processed[self.chunks_per_file:]
                                    self.save_batch(to_save, file_count)
                                    file_count += 1

                    batch_texts = []
            
            pbar.close()

        # Process remaining batch texts
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
                    pending_chunks.append(chunk_tokens)
                    current_chunk = current_chunk[L:]

        # Process remaining tokens in current_chunk
        if current_chunk:
            pending_chunks.append(current_chunk)
            
        # Process remaining pending chunks
        if pending_chunks:
            # Process in chunks of gpu_batch_size to avoid OOM if pending is huge
            for i in range(0, len(pending_chunks), gpu_batch_size):
                batch = pending_chunks[i:i+gpu_batch_size]
                batch_results = self.process_batch(batch, device)
                processed.extend(batch_results)
                chunk_count += len(batch_results)
        
        # Save remaining processed
        while len(processed) > 0:
            # If we have more than chunks_per_file, save chunks_per_file
            # If less, save all
            if len(processed) >= self.chunks_per_file:
                to_save = processed[:self.chunks_per_file]
                processed = processed[self.chunks_per_file:]
            else:
                to_save = processed
                processed = []
            
            self.save_batch(to_save, file_count)
            file_count += 1
            
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
