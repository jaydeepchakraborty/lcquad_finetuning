from lcquad_finetuning.util.util_lib import *

class LCQUADSFTDataset(Dataset):

    def __init__(self, data_file):

        df = pd.read_csv(data_file)
        self.questions = df["question"].astype(str).tolist()
        self.sparql = df["sparql"].astype(str).tolist()
        self.prompt_with_response = df["prompt_with_response"].astype(str).tolist()
        self.prompt_without_response = df["prompt_without_response"].astype(str).tolist()

    def __getitem__(self, idx):
        return {
            "question": self.questions[idx],
            "sparql": self.sparql[idx],
            "prompt_with_response": self.prompt_with_response[idx],
            "prompt_without_response": self.prompt_without_response[idx]
        }

    def __len__(self):
        return len(self.questions)