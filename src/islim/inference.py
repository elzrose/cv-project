
import torch


def translate_sign_keypoints(
    keypoints: torch.Tensor,
    models: dict,
    max_length: int = 128
) -> str:
    """
    Translate preprocessed ISL keypoints into English.

    Input:
        (1, 4, 94, 128) or (4, 94, 128)

    Channels:
        x, y, z, visibility
    """

    if keypoints.dim() == 3:
        keypoints = keypoints.unsqueeze(0)

    expected_shape = (1, 4, 94, 128)

    if tuple(keypoints.shape) != expected_shape:
        raise ValueError(
            f"Expected {expected_shape}, got {tuple(keypoints.shape)}"
        )

    device = models["device"]

    encoder = models["encoder"]
    t5_encoder = models["t5_encoder"]
    decoder_first = models["decoder_first"]
    decoder_step = models["decoder_step"]
    lm_head = models["lm_head"]
    tokenizer = models["tokenizer"]

    pad_token_id = models["pad_token_id"]
    eos_token_id = models["eos_token_id"]

    kp = keypoints.to(device=device, dtype=torch.float32)

    with torch.no_grad():

        # 1. Visual encoder
        vis = encoder(kp)

        # 2. T5 encoder
        mem = t5_encoder(vis)

        # 3. First decoder step
        start = torch.full(
            (1, 1),
            pad_token_id,
            dtype=torch.long,
            device=device
        )

        out1 = decoder_first(mem, start)

        hidden, sa_keys, sa_vals, ca_keys, ca_vals = out1

        logits = lm_head(hidden)
        next_tok = int(logits[0, -1].argmax(dim=-1).item())

        tokens = []

        if next_tok != eos_token_id:
            tokens.append(next_tok)

        # 4. Autoregressive greedy decoding
        for _ in range(1, max_length):

            if next_tok == eos_token_id:
                break

            tok_in = torch.full(
                (1, 1),
                next_tok,
                dtype=torch.long,
                device=device
            )

            out = decoder_step(
                mem,
                tok_in,
                sa_keys,
                sa_vals,
                ca_keys,
                ca_vals
            )

            logits, sa_keys, sa_vals, ca_keys, ca_vals = out

            next_tok = int(logits[0, -1].argmax(dim=-1).item())

            if next_tok == eos_token_id:
                break

            tokens.append(next_tok)

    return tokenizer.decode(
        tokens,
        skip_special_tokens=True
    ).strip().lower()
