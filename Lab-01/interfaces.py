class NgramConfig:
    """
    n: N-gram size
    k: Add-k smoothing factor
    """
    def __init__(self, n: int, k: float):
        self.n = n
        self.k = k