import torch

def test_mask_logic():
    block_size = 2
    seq_len = 10
    start_idx = 1
    
    positions = torch.arange(seq_len)
    block_indices = torch.full((seq_len,), -1, dtype=torch.long)
    
    if seq_len > start_idx:
        valid_pos = positions[start_idx:]
        block_indices[start_idx:] = (valid_pos - start_idx) // block_size
        
    print(f"Block indices: {block_indices}")
    
    # i attends to j if block[j] < block[i]
    # mask[i, j] = 1 if attend, 0 if mask
    attention_mask = (block_indices[None, :] < block_indices[:, None]).int()
    
    print("Attention Mask (0=Mask, 1=Attend):")
    print(attention_mask)
    
    # Check t1 (idx 2) attending to t0 (idx 1)
    # t0 block = 0. t1 block = 0.
    # 0 < 0 -> False (0)
    t1_sees_t0 = attention_mask[2, 1]
    print(f"t1 (idx 2) sees t0 (idx 1): {t1_sees_t0}")
    
    # Check t2 (idx 3) attending to t0 (idx 1)
    # t2 block = 1. t0 block = 0.
    # 0 < 1 -> True (1)
    t2_sees_t0 = attention_mask[3, 1]
    print(f"t2 (idx 3) sees t0 (idx 1): {t2_sees_t0}")

    # Check t2 (idx 3) attending to t2 (idx 3)
    # 1 < 1 -> False (0)
    t2_sees_t2 = attention_mask[3, 3]
    print(f"t2 (idx 3) sees t2 (idx 3): {t2_sees_t2}")

if __name__ == "__main__":
    test_mask_logic()
