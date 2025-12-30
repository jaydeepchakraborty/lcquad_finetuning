from lcquad_finetuning.util.util_lib import *

class LCQUADRLHFDataset(Dataset):
    def __init__(self, data_file):
        self.df = pd.read_csv(data_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "question": row["question"],
            "generated_sparql": row["generated_sparql"],
            "score": torch.tensor(row["score"], dtype=torch.float32),
        }