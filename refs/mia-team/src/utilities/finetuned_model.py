import torch


def load_arcface_finetuned_hf_model(hf_model, ckpt_path, device, final_in_features=2152):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    hf_model.final_in_features = final_in_features
    hf_model.load_state_dict(state, strict=True)

    print(f"Loaded ArcFace fine-tuned MiewID weights from checkpoint: {ckpt_path}")
    print(f"Loaded keys: {len(state)}")

    return hf_model


def load_finetuned_backbone_into_hf_model(hf_model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    candidate_prefixes = [
        "backbone.hf_model.",
        "module.backbone.hf_model.",
    ]

    filtered_state = None
    used_prefix = None

    for prefix in candidate_prefixes:
        tmp = {}
        for k, v in state.items():
            if k.startswith(prefix):
                new_k = k[len(prefix):]
                tmp[new_k] = v
        if len(tmp) > 0:
            filtered_state = tmp
            used_prefix = prefix
            break

    if filtered_state is None:
        raise ValueError(
            "Could not find fine-tuned backbone weights in checkpoint. "
            "Available example keys:\n" + "\n".join(list(state.keys())[:30])
        )

    missing, unexpected = hf_model.load_state_dict(filtered_state, strict=False)

    print(f"Loaded fine-tuned backbone from checkpoint: {ckpt_path}")
    print(f"Used prefix: {used_prefix}")
    print(f"Loaded keys: {len(filtered_state)}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("\nFirst few missing keys:", missing[:10])
    if len(unexpected) > 0:
        print("\nFirst few unexpected keys:", unexpected[:10])

    return hf_model

