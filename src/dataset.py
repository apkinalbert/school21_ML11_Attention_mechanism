from torch.utils.data import Dataset
import pandas as pd
import torch

class NamesDataset(Dataset):
    def __init__(
        self,
        latin_names: pd.Series,
        cyrillic_names: pd.Series,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.latin_names = torch.tensor(latin_names.to_list(), dtype=torch.long, device=device)
        self.cyrillic_names = torch.tensor(cyrillic_names.to_list(), dtype=torch.long, device=device)

    def __len__(self) -> int:
        return len(self.latin_names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        latin = self.latin_names[index]
        cyrillic = self.cyrillic_names[index]

        return latin, cyrillic