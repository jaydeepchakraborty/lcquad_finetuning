from lcquad_finetuning.util.util_lib import *

class LCQUADRMDataset(Dataset):

    def __init__(self, data_file):

        df = pd.read_csv(data_file)
        self.questions = df["question"].astype(str).tolist()
        self.original_sparql = df["original_sparql"].astype(str).tolist()
        self.generated_sparql = df["generated_sparql"].astype(str).tolist()
        self.reward_score = df["reward_score"].astype(float).tolist()

    def __getitem__(self, idx):
        return {
            "question": self.questions[idx],
            "original_sparql": self.original_sparql[idx],
            "generated_sparql": self.generated_sparql[idx],
            "reward_score": self.reward_score[idx]
        }

    def __len__(self):
        return len(self.questions)