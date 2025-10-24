from util.util_lib import *
from lcquad_finetuning.util.lcquad_util import LCQuadUtil

class LCQUADDataset(Dataset):

    def __init__(self, json_file, tokenizer):

        self.tokenizer = tokenizer
        df = pd.read_csv(json_file)
        self.data = df.apply(self.format_data, axis=1).tolist()

    def format_data(self, row):
        format_entry = LCQuadUtil.format_entry(row, "train")

        # Encode the text into a list of token IDs
        encoded_tokens = self.tokenizer.encode(format_entry)

        # Decode each token ID back into its original text chunk
        decoded_tokens = [self.tokenizer.decode([token_id]) for token_id in encoded_tokens]

        format_data = {
            "org_text": format_entry,
            "encoded_text": encoded_tokens,
            "decoded_text": decoded_tokens,
        }

        return format_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)