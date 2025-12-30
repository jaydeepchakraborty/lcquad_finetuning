from lcquad_finetuning.util.util_lib import *

class LCQUADCLMDataset(Dataset):

    def __init__(self, data_file_df, tokenizer, max_length=512):

        self.max_length = max_length
        self.sparql = data_file_df["sparql"].astype(str).tolist()

        self.enc = tokenizer(self.sparql,
                                truncation=True,
                                max_length=self.max_length,
                                padding=True,
                                return_tensors="pt")

    def __len__(self):
        return len(self.sparql)

    def __getitem__(self, idx):
        input_ids = self.enc["input_ids"].squeeze(0)
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone()
        }