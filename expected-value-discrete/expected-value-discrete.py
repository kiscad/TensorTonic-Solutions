import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.array(x)
    p = np.array(p)
    if abs(np.sum(p) - 1.0) > 1e-5:
        raise ValueError()
    return np.dot(x, p)
