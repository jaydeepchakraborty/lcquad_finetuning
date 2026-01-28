from lcquad_finetuning.util.util_lib import *

class LCQUADCLMDataset(Dataset):

    def __init__(self, data_file_df, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sparql = data_file_df["sparql"].unique() #.astype(str).tolist()

    def __len__(self):
        return len(self.sparql)

    def __getitem__(self, idx):

        text = self.sparql[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True, # attention mask is important for training purpose
        )

        # input_ids, attention_mask
        return encoding