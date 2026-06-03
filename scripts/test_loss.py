import torch
from src.models.loss import AdaptiveLossWeights

def test():
    losses = torch.tensor([0.02, 500.0, 1.0, 0.001, 0.0, 1.5e-10])
    layer = AdaptiveLossWeights(n_losses=6)
    w = layer(losses)
    print("weights:", w)

test()
