import torch
from torch import nn
import torch.nn.functional as F
from losses import focal_loss

def train_one_iter(model, optimizer, origin, inpaint, mask):
    model.train()
    pred = model(inpaint)
    pixel_loss = F.mse_loss(pred, mask)
    gernal_loss = F.mse_loss((1.-pred)*inpaint,(1.-mask) *origin)
    loss = pixel_loss + gernal_loss
    loss.backward()
    optimizer.step()
    return loss





if __name__ == "__main__":
    mask1 = torch.randn(3, 1, 256, 256)
    mask2 = torch.randn(3, 1, 256, 256)
    fl = focal_loss()

    print(fl(mask1, mask2))