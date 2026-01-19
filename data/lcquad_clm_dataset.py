from lcquad_finetuning.util.util_lib import *

class LCQUADCLMDataset(Dataset):

    def __init__(self, data_file_df, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sparql = data_file_df["sparql"].astype(str).tolist()

    def __len__(self):
        return len(self.sparql)

    def __getitem__(self, idx):

        text = self.sparql[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False
        )

        return {
            "input_ids": encoding["input_ids"]
        }