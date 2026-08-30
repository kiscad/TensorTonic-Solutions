import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return (
        torch.add(x, y).tolist() if op == "add"
        else torch.mul(x, y).tolist() if op == "multiply"
        else torch.matmul(x, y).tolist() if op == "matmul"
        else torch.pow(x, y).tolist() if op == "power"
        else torch.max(x, y) if op == "max"
        else None
    )