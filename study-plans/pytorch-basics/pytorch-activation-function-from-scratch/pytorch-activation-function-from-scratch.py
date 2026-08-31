import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    x = torch.tensor(x, dtype=torch.float32)
    if method == "relu":
        res = torch.clamp(x, min=0)
        # res = torch.nn.functional.relu(x)
    elif method == "sigmoid":
        res = 1.0 / (1.0 + torch.exp(-x))
        # res = torch.nn.functional.sigmoid(x)
    elif method == "tanh":
        e1 = torch.exp(x)
        e_ = torch.exp(-x)
        res = (e1 - e_) / (e1 + e_)
        # res = torch.nn.functional.tanh(x)
    elif method == "leaky_relu":
        res = torch.where(x>0, x, 0.01*x)
        # res = torch.nn.functional.leaky_relu(x)

    return res.tolist()