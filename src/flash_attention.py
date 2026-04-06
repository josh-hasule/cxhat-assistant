import torch 
import torch.nn.functional as F 


# Detection: Try to Load FA3 on Hopper+ GPUs 
def _load_flash_attention_3(): 
    """Try to load Flash Attention 3 (requiress Hopper GPU, sm90)."""
    if not torch.cuda.is_available(): 
        return None 
    try: 
        major, _ = torch.cuda.get_device_capability()
        if major != 9: 
            return None 
        import os 
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel 
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception: 
        return None 
    
_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None 

# Override for testing: set to 'fa3'
_override_impl = None 

def _resolve_use_fa3(): 
    """Decide once whether to use FA3, based on availability, override, and dtype."""
    if _override_impl == 'fa3': 
        assert HAS_FA3, "cannot override to FA3: not available on this hardware"
        return true 
    
    if _override_impl == "sdpa": 
        return False 
    if HAS_FA3: 
        # FA3 Hopper kernels only supoort bf16; fp16/fp32 must use SDPA fallback 
        from src.common import COMPUTE_DTYPE 
        if COMPUTER_DTYPE == torch.bfloat16: 
            return True 
        return False 
    return False 

USE_FA3 = _resolve_use_fa3()


# SDPA helpers 
def _sdpa_attention(q,k,v, window_size, enable_gpa): 
    """
    SDPA attention with sliding window support. 
    q, k, v are (B, H, T, D) format. 
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length 
    if (window < 0 or window >= Tq) and Tq == Tk: 
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gpa=enable_gpa)
    
    # Single token generation 
    if Tq == 1: 
        if window >=0 and window < Tk: 
            # window id "Left" tokens we need to include (window + 1) keys total 
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gpa=enable_gpa)
    
    # Need explicit masj for sliding window/chunk inference 
    device = q.device 
    # For chunk inference (Tq != Tk), is_causal in not aligned to cache position =? build an explicit boo mask 
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx 

    # sliding window (left)
    if window >= 0 and window < Tk: 
        mask = mask & ((row_idx - col_idx) <= window)
    
    return F.scaled_dot_product_attention(q, k, v, attn_mask=maks, enable_gpa=enable_gpa)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, 1)):: 
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if USE_FA3: 
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    
    # SDPA Fallback: tranpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gpa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gpa)
    return y.transpose(1, 2)

def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)

from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
