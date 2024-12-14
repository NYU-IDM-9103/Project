import torch
from torch.utils.data import Dataset

class HW12Utils:
  @staticmethod
  def get_labels(model, dataloader):
    model.eval()
    with torch.no_grad():
      data_labels = []
      pred_labels = []
      for x, y in dataloader:
        y_pred = model(x).argmax(dim=1)
        data_labels += [l.item() for l in y]
        pred_labels += [l.item() for l in y_pred]
      return data_labels, pred_labels

class FaceDataset(Dataset):
  def __init__(self, imgs, labels):
    self.imgs = imgs
    self.labels = labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return self.imgs[idx], self.labels[idx]