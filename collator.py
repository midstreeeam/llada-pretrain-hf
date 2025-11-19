import random
import torch
from typing import Dict, List, Any, Callable, Union
from transformers import PreTrainedTokenizer



class NTPCollator:
    """
    用于NTP（Next Token Prediction）预训练任务的数据整理器
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        text_key: str = 'text',
    ):
        """
        Args:
            tokenizer: 预训练的分词器
            max_length: 最大序列长度
            text_key: 如果指定，会从examples中提取该key对应的文本并转换为input_ids
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key
        
        # 获取特殊token的id
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id  # 添加BOS token
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        处理一个batch的数据
        
        Args:
            examples: 包含'input_ids'键或指定text_key的字典列表
            
        Returns:
            包含input_ids, attention_mask, labels的字典
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for example in examples:
            # 获取input_ids：优先使用现有的input_ids，否则从文本转换
            if 'input_ids' in example:
                input_ids = example['input_ids']
            elif self.text_key and self.text_key in example:
                # 从文本转换为input_ids
                text = example[self.text_key]
                encoded = self.tokenizer.encode(
                    text,
                    add_special_tokens=False,  # 手动添加BOS和EOS
                    truncation=True,
                    max_length=self.max_length - 2  # 留出BOS和EOS位置
                )
                input_ids = encoded
            else:
                raise ValueError(f"Example must contain either 'input_ids' or '{self.text_key}' key")
            
            # 确保序列以BOS开头，EOS结尾
            if input_ids[0] != self.bos_token_id:
                input_ids = [self.bos_token_id] + input_ids
            if input_ids[-1] != self.eos_token_id:
                input_ids = input_ids + [self.eos_token_id]
            
            # 确保序列长度不超过max_length
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
            
            # 创建NTP标签：直接使用input_ids，让模型前向处理移位
            labels = input_ids.copy()  # 直接复制，不做移位操作
            
            # 创建attention mask
            attention_mask = [1] * len(input_ids)
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
        
        # Padding处理
        batch_input_ids = self._pad_sequences(batch_input_ids, self.pad_token_id)
        batch_attention_mask = self._pad_sequences(batch_attention_mask, 0)
        batch_labels = self._pad_sequences(batch_labels, -100)
        
        return {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long),
            'return_dict': True,
        }
    
    def _pad_sequences(self, sequences: List[List[int]], pad_value: int) -> List[List[int]]:
        """
        对序列进行padding
        """
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = []
        
        for seq in sequences:
            padded_seq = seq + [pad_value] * (max_len - len(seq))
            padded_sequences.append(padded_seq)
        
        return padded_sequences
    






class LLaDACollator:
    """
    用于BERT预训练MLM任务的数据整理器
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        text_key: str = 'text',
        masking_strategy: str = "random", # random, semi_autoregressive, or semi_autoregressive_parallel
        block_size: int = 64,
    ):
        
        self.tokenizer = tokenizer
        
        self.max_length = max_length
        self.text_key = text_key  # 保存text_key
        self.masking_strategy = masking_strategy
        self.block_size = block_size

        # 获取特殊token的id
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.eos_token_id = tokenizer.eos_token_id  # 添加EOS token
        self.bos_token_id = tokenizer.bos_token_id  # 添加BOS token
        
        # 可用于随机替换的token范围
        self.vocab_size = tokenizer.vocab_size
        
        
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        处理一个batch的数据
        
        Args:
            examples: 包含'input_ids'键或指定text_key的字典列表
            
        Returns:
            包含input_ids, attention_mask, labels, token_change_labels的字典
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        batch_mlm_probs = []  # 添加用于存储每个样本MLM概率的列表
        
        for example in examples:
            # 获取input_ids：优先使用现有的input_ids，否则从文本转换
            if 'input_ids' in example:
                input_ids = example['input_ids']
            elif self.text_key and self.text_key in example:
                # 从文本转换为input_ids
                text = example[self.text_key]
                encoded = self.tokenizer.encode(
                    text, 
                    add_special_tokens=False,  # 我们手动添加BOS和结尾token
                    truncation=True,
                    max_length=self.max_length - 2  # 留出BOS和结尾token位置
                )
                input_ids = encoded
            else:
                raise ValueError(f"Example must contain either 'input_ids' or '{self.text_key}' key")
            

            if not input_ids or input_ids[0] != self.bos_token_id:
                input_ids = [self.bos_token_id] + input_ids

            input_ids.append(self.eos_token_id)


            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]


            if self.masking_strategy == "semi_autoregressive":
                masked_input_ids, labels, attention_mask, current_mlm_prob = self._mask_tokens_sar(input_ids)
            elif self.masking_strategy == "semi_autoregressive_parallel":
                masked_input_ids, labels, attention_mask, current_mlm_prob = self._mask_tokens_sar_parallel(input_ids)
            else:
                current_mlm_prob = self._get_mlm_probability()  # 为每个样本获取MLM概率
                # 创建MLM mask和labels
                masked_input_ids, labels = self._mask_tokens(input_ids, current_mlm_prob)
                # 创建attention mask
                attention_mask = [1] * len(masked_input_ids)
            
            batch_mlm_probs.append(current_mlm_prob)  # 添加到列表中

            batch_input_ids.append(masked_input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        
        # Padding处理
        batch_input_ids = self._pad_sequences(batch_input_ids, self.pad_token_id)
        batch_labels = self._pad_sequences(batch_labels, -100)

        # Handle attention masks based on strategy
        if self.masking_strategy == "semi_autoregressive_parallel":
            # For 2D attention masks, we need to pad them to match the padded sequence length
            max_seq_len = len(batch_input_ids[0])  # All sequences should be same length after padding

            # Pad each 2D attention mask to max_seq_len x max_seq_len
            padded_attention_masks = []
            for attn_mask in batch_attention_mask:
                current_len = len(attn_mask)
                if current_len < max_seq_len:
                    # Create a larger mask and copy the existing values
                    padded_mask = torch.zeros((max_seq_len, max_seq_len), dtype=torch.int)
                    padded_mask[:current_len, :current_len] = torch.tensor(attn_mask, dtype=torch.int)
                    padded_attention_masks.append(padded_mask)
                else:
                    padded_attention_masks.append(torch.tensor(attn_mask, dtype=torch.int))

            # Convert to 4D attention_bias format expected by the model: (batch_size, 1, seq_len, seq_len)
            attention_bias_tensor = torch.stack(padded_attention_masks, dim=0).unsqueeze(1).float()
            # Convert 1s to 0s and 0s to -inf for attention bias
            attention_bias_tensor = torch.where(attention_bias_tensor == 1, 0.0, float('-inf'))

            return {
                'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
                'attention_mask': torch.ones((len(batch_input_ids), max_seq_len), dtype=torch.long),  # Standard attention mask (no padding masking)
                'attention_bias': attention_bias_tensor,
                'labels': torch.tensor(batch_labels, dtype=torch.long),
                'current_mlm_prob': torch.tensor(batch_mlm_probs, dtype=torch.float),  # 使用列表转换
                'return_dict': True,
            }
        else:
            # For 1D attention masks, pad normally
            batch_attention_mask_padded = self._pad_sequences(batch_attention_mask, 0)
            attention_mask_tensor = torch.tensor(batch_attention_mask_padded, dtype=torch.long)

            return {
                'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
                'attention_mask': attention_mask_tensor,
                'labels': torch.tensor(batch_labels, dtype=torch.long),
                'current_mlm_prob': torch.tensor(batch_mlm_probs, dtype=torch.float),  # 使用列表转换
                'return_dict': True,
            }
    
    def _get_mlm_probability(self, eps: float = 1e-3) -> float:
        t = random.uniform(0, 1)
        # 这是一个简单的线性噪声调度
        p_mask = (1 - eps) * t + eps
        return p_mask

    def _mask_tokens_sar(self, input_ids: List[int]) -> tuple:
        """
        Semi-Autoregressive masking strategy.
        Randomly selects a block to mask, keeping history visible and future hidden.
        """
        labels = [-100] * len(input_ids)
        attention_mask = [1] * len(input_ids)
        
        # Calculate number of blocks
        # We don't want to mask BOS (idx 0), so effective length is len - 1
        # But simplicity: just chunk the whole sequence
        seq_len = len(input_ids)
        
        # Determine start index for blocks. Usually after BOS.
        start_idx = 1 if input_ids[0] == self.bos_token_id else 0
        
        effective_len = seq_len - start_idx
        if effective_len <= 0:
            return input_ids, labels, attention_mask, 0.0

        # Randomly select a split point
        # We want to split at: start_idx + k * block_size
        # max_blocks = math.ceil(effective_len / self.block_size)
        
        # We want to pick a target block index [0, max_blocks-1]
        # If we pick block k:
        #   History: 0 to start_idx + k * block_size (Visible)
        #   Target:  start_idx + k * block_size to start_idx + (k+1) * block_size (Masked & Labelled)
        #   Future:  start_idx + (k+1) * block_size to end (Masked & Hidden/Ignored)
        
        # However, training is more efficient if we just pick a random cut point?
        # No, sticking to block grid is better for consistency with inference.
        
        num_possible_blocks = (effective_len + self.block_size - 1) // self.block_size
        if num_possible_blocks == 0:
             # Sequence too short, just mask everything after BOS?
             target_block_idx = 0
        else:
             target_block_idx = random.randint(0, num_possible_blocks - 1)
             
        block_start = start_idx + target_block_idx * self.block_size
        block_end = min(start_idx + (target_block_idx + 1) * self.block_size, seq_len)
        
        # 1. Mask and Label the Target Block
        for i in range(block_start, block_end):
            labels[i] = input_ids[i]
            input_ids[i] = self.mask_token_id
            
        # 2. Mask and Hide the Future
        # Future starts at block_end
        for i in range(block_end, seq_len):
            input_ids[i] = self.mask_token_id # Mask future tokens
            attention_mask[i] = 0             # Hide future tokens from attention
            labels[i] = -100                  # Do not compute loss for future
            
        # 3. History (0 to block_start) remains untouched (input_ids correct, labels -100, attn 1)
        
        # Calculate effective mask probability for logging/scheduler (approximate)
        # This isn't really "MLM prob" but "Progress"
        current_progress = block_start / seq_len
        
        return input_ids, labels, attention_mask, current_progress

    def _mask_tokens_sar_parallel(self, input_ids: List[int]) -> tuple:
        """
        Semi-Autoregressive Parallel masking strategy.
        Uses block-causal attention mask to train all blocks in parallel.
        Each block can only attend to previous blocks and itself.
        """

        seq_len = len(input_ids)
        
        # Convert input_ids to tensor for efficient operations
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)

        # Determine start index for blocks. Usually after BOS.
        start_idx = 1 if input_ids_tensor[0] == self.bos_token_id else 0
        
        # Create block indices
        positions = torch.arange(seq_len)
        block_indices = torch.full((seq_len,), -1, dtype=torch.long)
        if seq_len > start_idx:
            valid_pos = positions[start_idx:]
            block_indices[start_idx:] = (valid_pos - start_idx) // self.block_size

        # Determine max block index
        max_block_idx = block_indices.max().item()
        
        # Checkerboard Masking: Randomly choose to mask Even or Odd blocks
        # This allows training 50% of blocks in parallel.
        # Masked blocks predict targets. Unmasked blocks provide context.
        parity = torch.randint(0, 2, (1,)).item()
        
        masked_input_ids = input_ids_tensor.clone()
        labels = input_ids_tensor.clone()
        
        # Create mask for positions that should be masked (matching parity)
        # Ignore -1 (BOS/special tokens)
        mask_condition = (block_indices != -1) & (block_indices % 2 == parity)
        
        # Apply masking
        masked_input_ids[mask_condition] = self.mask_token_id
        
        # Set labels: -100 for unmasked (context) tokens and special tokens
        # We only train on the masked tokens
        labels[~mask_condition] = -100
        
        # Also ensure BOS is -100 (already handled by ~mask_condition since block_idx=-1)
        # But explicitly for safety if logic changes
        labels[0] = -100

        # Create 2D attention mask
        # i attends to j if block[j] < block[i]
        # AND we must allow attending to context (unmasked) blocks.
        # The original logic (block[j] < block[i]) handles this:
        # If Block i is Masked (e.g. 2), it attends to Block 1 (Unmasked). 1 < 2.
        # If Block i is Masked (e.g. 2), it does NOT attend to Block 2. 2 < 2 False.
        # So the logic remains valid.
        
        # Vectorized attention mask creation
        # shape: (seq_len, seq_len)
        # 1 means attend, 0 means mask (will be converted to -inf)
        attention_mask = (block_indices[None, :] < block_indices[:, None]).int()
        
        # Allow attending to special tokens (block_idx = -1)
        # Any token should attend to BOS (idx 0)
        # block_indices[0] is -1. block_indices[i] >= 0.
        # -1 < 0. True. So BOS is visible.
        
        # What about BOS attending to itself?
        # -1 < -1 False.
        # So BOS cannot see itself?
        # We should allow BOS to see itself.
        # And generally, special tokens should see themselves?
        # Or maybe it doesn't matter since we don't predict BOS.
        # But for stability, let's allow diagonal for special tokens.
        special_tokens_mask = (block_indices == -1)
        attention_mask[special_tokens_mask, special_tokens_mask] = 1

        # For logging/scheduler compatibility
        # This is an approximation of how much of the sequence is being predicted
        current_mlm_prob = mask_condition.float().mean().item() if mask_condition.any() else 0.0

        return masked_input_ids.tolist(), labels.tolist(), attention_mask.tolist(), current_mlm_prob

    def _mask_tokens(self, input_ids: List[int], current_mlm_prob ) -> tuple:
        """
        对输入序列进行MLM masking
        
        Args:
            input_ids: 输入token序列
            
        Returns:
            (masked_input_ids, labels, token_change_labels): mask后的序列、MLM标签和token变化标签
        """
        
        labels = [-100] * len(input_ids)  
  
        maskable_positions = []

        for i, token_id in enumerate(input_ids):
            if token_id not in [self.cls_token_id, self.pad_token_id, self.bos_token_id, self.sep_token_id]:  # BOS也不能被mask
                maskable_positions.append(i)
                
        
        # 强制处理结尾token（EOS或SEP）
        # if end_token_position is not None:
        #     labels[end_token_position] = input_ids[end_token_position]  # 保存原始结尾token作为标签
            
        #     input_ids[end_token_position] = self.mask_token_id
                
        #     if end_token_position in maskable_positions:
        #         maskable_positions.remove(end_token_position)
        
        # 随机选择其他需要mask的位置
        num_to_mask = max(1, int(len(maskable_positions) * current_mlm_prob))
        masked_positions = random.sample(maskable_positions, 
                                       min(num_to_mask, len(maskable_positions)))
        
        for pos in masked_positions:
            labels[pos] = input_ids[pos]  # 保存原始token作为标签
            input_ids[pos] = self.mask_token_id
               

        return input_ids, labels
    
    def _pad_sequences(self, sequences: List[List[int]], pad_value: int) -> List[List[int]]:
        """
        对序列进行padding
        """
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = []
        
        for seq in sequences:
            padded_seq = seq + [pad_value] * (max_len - len(seq))
            padded_sequences.append(padded_seq)
        
        return padded_sequences
