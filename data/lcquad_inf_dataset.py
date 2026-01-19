from lcquad_finetuning.util.util_lib import *

class LCQUADINFDataset(Dataset):

    def __init__(self, data_file):

        df = pd.read_csv(data_file)
        self.questions = df["question"].astype(str).tolist()
        self.original_sparql = df["sparql"].astype(str).tolist()
        self.prompt_without_response = df["prompt_without_response"].astype(str).tolist()

    def __getitem__(self, idx):
        return {
            "question": self.questions[idx],
            "original_sparql": self.original_sparql[idx],
            "prompt_without_response": self.prompt_without_response[idx]
        }

    def __len__(self):
        return len(self.questions)