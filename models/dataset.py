from torch.utils.data import Dataset
import torch


class PairedDataset(Dataset):
    """
    Expects 'data' to be a list (or array-like) of length N,
    where each item is (gcms_vector, smell_vector).

    Each vector could be:
      - a NumPy array of shape [feature_dim]
      - a Python list
      - etc.
    We'll just return them as Tensors.
    """

    def __init__(self, data):
        self.data = data  # data = [(gcms_vec, smell_vec), (gcms_vec, smell_vec), ...]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gcms_vec, smell_vec = self.data[idx]

        # Convert to torch.FloatTensors (if they aren't already)
        gcms_vec = torch.tensor(gcms_vec, dtype=torch.float)
        smell_vec = torch.tensor(smell_vec, dtype=torch.float)

        return gcms_vec, smell_vec
