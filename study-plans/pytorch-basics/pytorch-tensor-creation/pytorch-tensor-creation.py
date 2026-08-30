import torch

def create_tensor(method, shape, value=0.0):
    """
    Returns: list
    """
    return (torch.zeros(shape).tolist() if method == "zeros"
            else torch.ones(shape).tolist() if method == "ones"
            else torch.full(shape, value).tolist() if method == "full"
            else None
           )