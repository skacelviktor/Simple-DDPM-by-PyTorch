import torch
import torch.nn as nn

def count_parameters(model: nn.Module, detailed: bool = True):
    total_params = 0
    details = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_count = param.numel()
        total_params += param_count
        if detailed:
            details.append((name, param_count))

    if detailed:
        print(f"{'Module':<60} {'Parameters':>12}")
        print("=" * 75)
        module_summaries = {}
        for name, count in details:
            module_root = name.split('.')[0]
            module_summaries[module_root] = module_summaries.get(module_root, 0) + count

        for module, count in module_summaries.items():
            print(f"{module:<60} {count:>12,}")

        print("=" * 75)

    print(f"{'Total Trainable Parameters':<60} {total_params:>12,}")
    return total_params