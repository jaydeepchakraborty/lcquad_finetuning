from lcquad_finetuning.util.util_lib import *

class LCQUADSFTDataset(Dataset):

    def __init__(self, data_file):

        df = pd.read_csv(data_file)
        self.questions = df["question"].astype(str).tolist()
        self.sparql = df["sparql"].astype(str).tolist()
        self.entry = df["entry"].astype(str).tolist()

    def __getitem__(self, idx):
        return {
            "question": self.questions[idx],
            "sparql": self.sparql[idx],
            "entry": self.entry[idx]
        }

    def __len__(self):
        return len(self.questions)