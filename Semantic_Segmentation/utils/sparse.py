import math
import torch
import MinkowskiEngine as ME

def dense2sparse(
    inputs: torch.Tensor,    # (B, 4, D, H, W)
    targets: torch.Tensor,   # (B, D, H, W)
    device: torch.device,
    eps: float = 1e-8
) -> (ME.SparseTensor, torch.Tensor):
    B, C, D, H, W = inputs.shape
    assert C == 4, "0: occ, 1~3: RGB"
    L = D * H * W

    # 1) flatten spatial dims
    flat_in  = inputs.view(B, C, L)     # (B, 4, L)
    flat_lbl = targets.view(B, 1, L)    # (B, 1, L)

    # 2) occupancy mask
    mask     = (flat_in[:, 0].abs() > eps)  # (B, L)

    # 3) non-zero
    batch_idx, flat_idx = mask.nonzero(as_tuple=True)  # 둘 다 (N,)

    # 4) recover 3D coords
    z = flat_idx // (H * W)
    rem = flat_idx %  (H * W)
    y = rem // W
    x = rem %  W

    # 5) build coords
    coords = torch.stack([batch_idx, z, y, x], dim=1) \
                  .to(torch.int64) \
                  .to(device)               # (N,4)

    # 6) single global index
    global_idx = batch_idx * L + flat_idx      # (N,)

    # 7) gather features
    flat_feats = flat_in.permute(0, 2, 1).reshape(-1, C)  # (B*L, C)
    feats = flat_feats[global_idx].contiguous().to(device)  # (N, C)

    # 8) gather labels
    flat_labels = flat_lbl.view(B * L)                   # (B*L,)
    lbls = flat_labels[global_idx].contiguous().to(device)  # (N,)

    # 9) SparseTensor
    sparse = ME.SparseTensor(
        features    = feats,
        coordinates = coords,
        device      = device,
    )
    return sparse, lbls

def measure_sparse_flops(model: ME.MinkowskiNetwork, sparse_input: ME.TensorField) -> int:
    model.eval()
    total_flops = 0
    handles = []

    def conv_hook(module, inputs, output):
        nonlocal total_flops
        x = inputs[0]
        N = x.F.shape[0]
        w = getattr(module, 'kernel', None)
        if w is None or not hasattr(w, 'shape'):
            raise RuntimeError(f"No kernel on: {module}")

        shape = list(w.shape)
        
        if shape[0] == module.out_channels and shape[1] == module.in_channels:
            Cout, Cin = shape[0], shape[1]
            kernel_size = shape[2:]
        elif shape[1] == module.out_channels and shape[0] == module.in_channels:
            Cin, Cout = shape[0], shape[1]
            kernel_size = shape[2:]
        else:
            kernel_size = [shape[0]]
            Cin, Cout = shape[1], shape[2]

        kv = int(math.prod(kernel_size))
        total_flops += N * (Cin * kv * Cout * 2)

    def lin_hook(module, inputs, output):
        nonlocal total_flops
        x = inputs[0]
        feat = getattr(x, 'F', x)
        N = feat.shape[0]

        if hasattr(module, 'kernel') and isinstance(module.kernel, torch.Tensor):
            w = module.kernel
        elif hasattr(module, 'linear') and hasattr(module.linear, 'weight'):
            w = module.linear.weight
        elif hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
            w = module.weight
        else:
            raise RuntimeError(f"No weight found in linear module: {module}")

        Cout, Cin = w.shape[0], w.shape[1]
        total_flops += N * (Cin * Cout * 2)

    # Register hooks on all conv/trans­pose‐conv and linear layers
    for m in model.modules():
        if isinstance(m, (
            ME.MinkowskiConvolution,
            ME.MinkowskiConvolutionTranspose,
            ME.MinkowskiGenerativeConvolutionTranspose,
        )):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, ME.MinkowskiLinear):
            handles.append(m.register_forward_hook(lin_hook))

    with torch.no_grad():
        _ = model(sparse_input)

    for h in handles:
        h.remove()

    return total_flops