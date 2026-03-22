"""
Minimal FP8 training for nanochat - tensorwise dynamic scaling only. 

Drop-in replacement for torchao's Float8Linear (~2000 lines) with ~150 lines. 
We only need the "tensorwise" recipe (one scalar scale per tensor), not the full 
generality of torchao (rowwise scaling, FSDP float8 all-gather, DTensor, tensor subclass dispatch tables, etc.)

How FP6 training works 
======================
A standard Linear layer does one matmul in forward and two in backward: 
    forward:        output      = input @ weight.T 
    backward:       grad_input  = grad_output @ weight 
                    grad_weight = grad_output.T @ input 

FP8 training wraps each of these three matmuls with: 
    1. Compute scale = FP8_MAX / max(|tensor|) for each operand 
    2. Quantize: fp8_tensor = clamp(tensor * scale, -FP8_MAX, FP8_MAX).to(fp8)
    3. Matmul via torch._scaled_mm (cuBLAS FP8 kernel, ~2x faster than bf16)
    4. Dequantize: _scaled_mm handles this internally using the inverse scales 

The key insight: torch._scaled_mm and the float8 dtypes are Pytorch built-ins. 
torchao is just orchestration around these primitives. We can call them directly. 

FP8 dtype choice 
=================
There are two FP8 formats. We use both, following the standard convention: 
    - float8_e4m3fn: 4-bit exponent, 3-bit mantissa, range [-448, 448]
      Higher precision (more mantissa bits), used for input and weight. 
    - float8_e5m2: 5-bit exponent, 2-bit mantissa, range [-57344, 57344]
      Wider range (more exponent bits), used for gradients which can be large.

torch._scaled_mm layout requirements 
=====================================
"""

import torch 
import torch.nn as nn 

from src.common import COMPUTE_DTYPE 

# Avoid division by zero when computing scale from an all-zeros tensor 
EPS = 1e-12

@torch.no_grad()
def _to_fp8(x, fp8_dtype): 
    """Dynamically quantize a tensor to FP8 using tensorwise scaling. 

    "Tensorwise" means one scalar scale for the entire tensor (as opposed to "rowwise" which computes a seperate scale per row). Tensorwise 
    is faster because cuBLAS handles the scaling; rowwise needs the CUTLASS kernel. 

    Find the biggest value -> calculate shrink factor -> shrink everything -> return 8-bit version 

    Returns (fp8_data, inverse_scale) for use with torch._scaled_mm. 
    """
    fp8_max = torch.finfo(fp8_dtype).max 
    # Compute the max absolute value across the entire tensor 
    amax = x.float().abs().max()
    # Scale maps [0, amax] -> [0, fp8_max]. Use float64 for the division to 
    # ensure consistent numerics between torch.compile and eager mode. 
    # (torchao does the same upcast - without it, compile/eager can diverge)
    scale = fp8_max / amax.double().clamp(min=EPS)
    scale = scale.float()
    # Quantize: scale into FP8 range, saturate (clamp prevents overflow when casting) - 
    # PyTorch's default is to wrap, not saturate), then cast to FP8
    x_scaled = x.float() * scale 
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max)
    x_fp8 = x_clamped.to(fp8_dtype)
    # _scaled_mm expects the *inverse* of our scale (it multiplies by this to convert FP8 values
    # back to the original range during the matmul) 
    inv_scale = scale.reciprocal()
    return x_fp8, inv_scale 

def _to_col_major(x): 
    """Rearrange a 2D tensor's memory to column-major layout. 

    torch._scaled_mm requires its second operand in column-major layout. 
    The trick: transpose -> contiguous (forces a copy in transposed order)
    -> transpose back. The result has the same logical shape but colum-major strides 
    e.g. a [M, N] tensor gets strides (1, M) instead if (N, 1). 
    """
    return x.t().contiguous().t()

# allow_in_graph tells torch.compile to treat this as an opaque operation - 
# dynamo won't try to decompose it into smaller ops, see the module docstring 
# for how this differs from torchao's tensor subclass approach. 
@torch._dynamo.allow_in_graph 
class _Float8Matmul(torch.autograd.Function): 
    """Custom autograd for the three FP8 GEMMs for a Linear layer. 

    The forward quantizes input and weight to FP8 and saves 
    the qunatized tensors + scales for backward. 
    """

    @staticmethod 
    def forward(ctx, input_2d, weight): 
        # Quantize both operands to e4m3 (higher precision format)
        input_fp8, input_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)
        ctx.save_for_backward(input_fp8, input_inv, weight_fp8, weight_inv)

        # output = input @ weight.T 
        # input_fp8 is [B, K] contiguous = row-major (good for first arg)
        # weight_fp8 is [N, K] contiguous, so weight_fp8.t() is [K, N] with 
        # strides (1, K) = column-major (good for second arg, no copy needed!)
        output = torch._scaled_mm(
            input_fp8, 
            weight_fp8.t(), 
            scale_a=input_inv,
            scale_b=weight_inv, 
            out_dtype=input_2d.dtype, 
            # use_fast_accum=True accumulates the dot products in lower precision. 
            # slightly less accurate but measurably faster. Standard practice for 
            # the forward pass; we use False in backward for more precise gradients. 
            use_fast_accum=True,
        )

        return output 

    @staticmethod
    def backward(ctx, grad_output): 
        in_fp8, in_inv, w_fp8, w_inv = ctx.saved_tensors 

        # === GEMM 1: grad_input = grad_output @ weight === 
        # Shapes: [B, N] @ [N, K] -> [B, K]
        # Gradients use e5m2 (wider range), weights use e4m3 (higer precision)
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)
        # go_fp8 is [B, N] contiguous = row-major, good for first arg 
        # w_fp8 is [N, K] contiguous = row-major, need column-major for second arg 
        w_col = _to_col_major(w_fp8)
        grad_input = torch._scaled_mm(
            go_fp8, 
            w_col, 
            scale_a=go_inv, 
            scale_b=w_inv, 
            out_dtype=grad_output.dtype, 
            use_fast_accum=False, 
        )

        # === GEMM 2: grad_weight = grad_output.T @ input === 
        # Shapes: [N, B] @ [B, K] -> [N, K]
        # go_fp8 is [B, N] contiguous, we need go.T = [N, B] as first arg. 
        # Transposing gives column-major, but first arg needs row-major, 
        # so we must call .contiguous() to physically rearrange the memory. 
        go_T = go_fp8.t().contiguous() # [N, B] row-major 
        in_col = _to_col_major(in_fp8) # [B, K] column-major 
        grad_weight = torch._scaled_mm(
            go_T, 
            in_col, 
            scale_a=go_inv, 
            scale_b=in_inv, 
            out_dtype=grad_output.dtype, 
            use_fast_accum=False, 
        )

        return grad_input, grad_weight 
    
class Float8Linear(nn.Linear): 
    """
    Drop-in nn.Linear replacement that does FP8 compute. 

    Weights and biases remain in their original precision (e.g fp32/bf16). 
    Only the matmul is performed in FP8 via the _Float8Matmul autograd function. 
    """

    def forward(self, input): 
        # Cast input to COMPUTE_DTYPE (typically bf16) since _scaled_mm expects
        # reduced precision input, we no longer rely on autocast to do this. 
        input = input.to(COMPUTE_DTYPE)
        # _scaled_mm only works on 2D tensors, so flatten batch dimensions 
        orig_shape = input.shape 
        input_2d = input.reshape(-1, orig_shape[-1])
        output = _Float8Matmul.apply(input_2d, self.weight)
        output = output.reshape(*orig_shape[:-1], output.shape[-1])
        if self.bias is not None: 
            output = output + self.bias.to(output.dtype)
        return output 
    
    @classmethod
    def from_float(cls, mod): 
        """Create Float8Linear from nn.Linear, sharing the same weight and bias. 

        Uses meta devies to avoid allocating a temporary weigh tensor - we create the module shell on meta (shapes/dtypes only, no memory), 
        the point .weight and .bias to the original module's parameters. 
        """
        with torch.device("meta"): 
            new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight 
        new_mod.bias  = mod.bias 
        return new_mod

class Float8LinearConfig: 
    """Minimal config matching torchao's API. Only tensorwise recipe is supported."""

    @staticmethod
    def from_recipe_namae(recipe_name): 
        """Replace nn.Linear layers with Float8Linear throughput a module. 

        Walks the module tree in post-order (children before parents) and swaps
        each nn.Linear that passes the optional filter. The new Float8Linear shares 
        the original weight and bias tensors - no copies, no extra memory. 

        Args: 
            module: Root module to convert. 
            config: Float8LinearConfig (accepted for API compat, only tensorwise supported). 
            module_filter_fn: Optional filter(module, fqn) -> bool. Only matching Linears 
            are converted. Common use: skip layers with dims divisible by 16
            (hardware requirement for FP8 matmuls on H1OO). 
        """
        def _convert(mod, prefix=""): 
            for name, child in mod.named_children(): 
                fqn = f"{prefix}.{name}" if prefix else name 
                _convert(child, fqn)
                if isinstance(child, nn.Linear) and not isinstance(child, Float8Linear): 
                    if module_filter_fn is None or module_filter_fn(child, fqn): 
                        setattr(mod, name, Float8Linear.from_float(child)) 
        
        _convert(module)
        return module 




if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
        print("CUDA capability:", torch.cuda.get_device_capability(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Small toy model ────────────────────────────────────────────────
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, 32)
            self.fc3 = nn.Linear(32, 10)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.fc3(x)
            return x

    # ────────────────────────────────────────────────────────────────

    torch.manual_seed(1234)

    # Create normal model
    model_fp = TinyNet().to(device=device, dtype=dtype)

    # Create FP8 version (replace all linears)
    model_fp8 = TinyNet().to(device=device, dtype=dtype)

    # Replace layers (we use the code's from_float method)
    def replace_with_fp8(mod):
        for name, child in mod.named_children():
            if isinstance(child, nn.Linear):
                fp8_layer = Float8Linear.from_float(child)
                setattr(mod, name, fp8_layer)
            else:
                replace_with_fp8(child)

    replace_with_fp8(model_fp8)

    # Make sure they have the same weights (sharing is intentional)
    print("\nWeight sharing check (should be very close to 1.0):")
    with torch.no_grad():
        diff = (model_fp.fc1.weight - model_fp8.fc1.weight).abs().max()
        print("fc1 weight max diff:", diff.item())

    # ────────────────────────────────────────────────────────────────
    # Test forward + backward
    # ────────────────────────────────────────────────────────────────

    x = torch.randn(4, 64, device=device, dtype=dtype) * 0.2   # small values → less saturation

    # Normal version
    model_fp.train()
    out_fp = model_fp(x)
    loss_fp = out_fp.pow(2).mean()          # dummy loss
    loss_fp.backward()

    grad_fp = model_fp.fc1.weight.grad.clone()

    # FP8 version
    model_fp8.train()
    out_fp8 = model_fp8(x)
    loss_fp8 = out_fp8.pow(2).mean()
    loss_fp8.backward()

    grad_fp8 = model_fp8.fc1.weight.grad.clone()

    # ────────────────────────────────────────────────────────────────
    # Compare
    # ────────────────────────────────────────────────────────────────

    print("\nForward pass comparison:")
    with torch.no_grad():
        diff_out = (out_fp - out_fp8).abs()
        print("Output max abs diff   :", diff_out.max().item())
        print("Output mean abs diff  :", diff_out.mean().item())
        print("Output L2 relative    :",
              torch.norm(out_fp - out_fp8) / (torch.norm(out_fp) + 1e-8))

    print("\nGradient comparison (fc1.weight):")
    with torch.no_grad():
        diff_grad = (grad_fp - grad_fp8).abs()
        print("Grad max abs diff     :", diff_grad.max().item())
        print("Grad mean abs diff    :", diff_grad.mean().item())
        print("Grad cosine sim       :",
              F.cosine_similarity(grad_fp.flatten(), grad_fp8.flatten(), dim=0).item())

    print("\nNaN / Inf check:")
    print("Normal model has NaN in grad? ", torch.isnan(grad_fp).any().item())
    print("FP8 model has NaN in grad?    ", torch.isnan(grad_fp8).any().item())

    print("\nDone. If you see:")
    print("  • small-ish output differences (1e-3 ~ 1e-1 range is common with tensorwise FP8)")
    print("  • gradients that are not zero / NaN / Inf")
    print("  → the code is most likely working correctly.")