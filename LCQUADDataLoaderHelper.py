from tinycss2 import tokenizer

from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.LCQUADDataset import LCQUADDataset

class LCQUADDataLoaderHelper:

    def __init__(self, tokenizer, config: dict):
        self.conf = config
        self.tokenizer = tokenizer

    def customized_collate_fn(
            self,
            batch,
            device="cpu"
    ):

        ignore_index = self.conf['model']['gpt_config']['basic_config']['ignore_index']
        allowed_max_length = self.conf['model']['gpt_config']['basic_config']['allowed_max_length']

        # ID for <SPARQL> special token
        sparql_token_id = self.tokenizer.convert_tokens_to_ids("<SPARQL>")
        pad_token_id = self.tokenizer.convert_tokens_to_ids("<PAD>")
        eod_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")

        # Find the longest sequence in the batch
        batch_max_length = max(len(item['encoded_text']) + 1 for item in batch)

        # Pad and prepare inputs and targets
        org_txt = []
        ip_encoded_text_lst, ip_modf_encoded_text_lst = [], []
        ip_encoded_tokens_lst, ip_modf_encoded_tokens_lst = [], []
        trgt_encoded_text_lst, trgt_modf_encoded_text_lst = [], []
        trgt_encoded_tokens_lst, trgt_modf_encoded_tokens_lst = [], []

        for item in batch:

            # ---------- 0. ORIGINAL TEXT FOR DEBUGGING ----------
            org_txt.append(item['org_text'])

            # ---------- 1. INPUT TOKENS ----------
            # ---------- 1a. INPUT TOKENS (Original) ----------
            ip_encoded_tokens = item['encoded_tokens'].copy()
            ip_encoded_tokens_lst.append(ip_encoded_tokens)

            # ---------- 1a. INPUT TOKENS (Modified) ----------
            ip_encoded_modf_tokens = item['encoded_tokens'].copy()
            ip_encoded_modf_tokens.append(eod_token_id) # Add an <|endoftext|> token

            # Pad sequences to max_length
            ip_encoded_modf_tokens = (
                    ip_encoded_modf_tokens + [pad_token_id] *
                    (batch_max_length - len(ip_encoded_modf_tokens))
            )
            # Converting to Tensor
            ip_encoded_modf_tokens = torch.tensor(ip_encoded_modf_tokens)
            #  OPTIONAL TRUNCATION (Optionally truncate to maximum sequence length)
            if allowed_max_length is not None:
                ip_encoded_modf_tokens = ip_encoded_modf_tokens[:allowed_max_length]
            ip_modf_encoded_tokens_lst.append(ip_encoded_modf_tokens)

            # ---------- 2. INPUT TEXT (DEBUG) ----------
            # ---------- 2a. INPUT TEXT (Original) (DEBUG) ----------
            ip_encoded_txt = item['encoded_text'].copy()
            ip_encoded_text_lst.append(ip_encoded_txt)

            # ---------- 2b. INPUT TEXT (Modified) (DEBUG) ----------
            ip_encoded_modf_txt = item['encoded_text'].copy()
            ip_encoded_modf_txt += ["<|endoftext|>"] # Add an <|endoftext|> token
            # Pad sequences to max_length
            ip_encoded_modf_txt = (
                    ip_encoded_modf_txt + ["<PAD>"] *
                    (batch_max_length - len(ip_encoded_modf_txt))
            )
            #  OPTIONAL TRUNCATION (Optionally truncate to maximum sequence length)
            if allowed_max_length is not None:
                ip_encoded_modf_txt = ip_encoded_modf_txt[:allowed_max_length]
            ip_modf_encoded_text_lst.append(ip_encoded_modf_txt)

            # ---------- 3. TARGET TOKENS ----------
            # ---------- 3a. TARGET TOKENS (Original) ----------
            # SHIFTED TARGET TOKENS (IDS)
            trgt_encoded_tokens = torch.tensor(ip_encoded_tokens).clone()
            trgt_encoded_tokens[:-1] = trgt_encoded_tokens[1:].clone() # Shift: targets to right by one level
            trgt_encoded_tokens[-1] = ignore_index  # last label undefined
            trgt_encoded_tokens_lst.append(trgt_encoded_tokens)

            # ---------- 3b. TARGET TOKENS (Modified) ----------
            trgt_encoded_modf_tokens = torch.tensor(ip_encoded_modf_tokens).detach().clone()
            trgt_encoded_modf_tokens[:-1] = trgt_encoded_modf_tokens[1:].clone() # Shift: targets to right by one level
            trgt_encoded_modf_tokens[-1] = ignore_index  # last label undefined

            # find position of <SPARQL>, MASK EVERYTHING BEFORE <SPARQL>
            try:
                idx = (trgt_encoded_modf_tokens == sparql_token_id).nonzero(as_tuple=True)[0].item()
            except:
                idx = len(trgt_encoded_modf_tokens)  # no <SPARQL> (shouldn't happen)

            # mask in ID targets
            trgt_encoded_modf_tokens[: idx + 1] = ignore_index  # mask question + <SPARQL>
            trgt_encoded_modf_tokens[trgt_encoded_modf_tokens == pad_token_id] = ignore_index  # mask padded area
            #  OPTIONAL TRUNCATION (Optionally truncate to maximum sequence length)
            if allowed_max_length is not None:
                trgt_encoded_modf_tokens = trgt_encoded_modf_tokens[:allowed_max_length]
            trgt_modf_encoded_tokens_lst.append(trgt_encoded_modf_tokens)

            # ---------- 4. TARGET TEXT ----------
            # ---------- 4a. TARGET TEXT (Original) ----------
            # SHIFTED TARGET TEXT
            trgt_encoded_txt = item['encoded_text'].copy()
            trgt_encoded_txt[:-1] = trgt_encoded_txt[1:] # Shift: targets to right by one level
            trgt_encoded_txt[-1] = "IGNORE"  # last label undefined
            trgt_encoded_text_lst.append(trgt_encoded_txt)

            # ---------- 4b. TARGET TEXT (Modified) ----------
            trgt_encoded_modf_txt = item['encoded_text'].copy()
            trgt_encoded_modf_txt[:-1] = trgt_encoded_modf_txt[1:] # Shift: targets to right by one level
            trgt_encoded_modf_txt[-1] = "IGNORE"  # last label undefined

            ## find position of <SPARQL>, MASK EVERYTHING BEFORE <SPARQL>
            try:
                idx_txt = trgt_encoded_modf_txt.index("<SPARQL>")
            except:
                idx_txt = len(trgt_encoded_modf_txt)

            for i in range(idx_txt + 1):
                trgt_encoded_modf_txt[i] = "IGNORE"

            trgt_encoded_modf_txt = ["IGNORE" if tok == "<PAD>" else tok for tok in trgt_encoded_modf_txt]
            #  OPTIONAL TRUNCATION (Optionally truncate to maximum sequence length)
            if allowed_max_length is not None:
                trgt_encoded_modf_txt = trgt_encoded_modf_txt[:allowed_max_length]
            trgt_modf_encoded_text_lst.append(trgt_encoded_modf_txt)

        # Convert list of inputs and targets to tensors and transfer to target device
        ip_modf_encoded_tokens_stacked_lst = torch.stack(ip_modf_encoded_tokens_lst).to(device)
        trgt_modf_encoded_tokens_stacked_lst = torch.stack(trgt_modf_encoded_tokens_lst).to(device)

        data_batch = {
            "org_txt": org_txt,
            "ip_encoded_text": ip_encoded_text_lst,
            "ip_modf_encoded_text": ip_modf_encoded_text_lst,
            "ip_encoded_tokens": ip_encoded_tokens_lst,
            "ip_modf_encoded_tokens": ip_modf_encoded_tokens_stacked_lst,
            "trgt_encoded_text": trgt_encoded_text_lst,
            "trgt_modf_encoded_text": trgt_modf_encoded_text_lst,
            "trgt_encoded_tokens": trgt_encoded_tokens_lst,
            "trgt_modf_encoded_tokens": trgt_modf_encoded_tokens_stacked_lst
        }

        return data_batch

    def load_dataloader(self, dataset_ind, dataset_file_path):

        num_workers = self.conf['model']['num_workers']
        if dataset_ind == "train":
            batch_size = self.conf['model']['batch_size']['train_batch_size']
        elif dataset_ind == "val":
            batch_size = self.conf['model']['batch_size']['val_batch_size']
        elif dataset_ind == "test":
            batch_size = self.conf['model']['batch_size']['test_batch_size']
        else:
            raise NotImplementedError

        print(f"Loading dataloader from {dataset_file_path}")
        with torch.serialization.safe_globals([LCQUADDataset]):
            dataset = torch.load(dataset_file_path, weights_only=False)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.customized_collate_fn,
            shuffle=False,
            num_workers=num_workers
        )

        return dataloader