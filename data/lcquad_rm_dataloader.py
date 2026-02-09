from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer
from lcquad_finetuning.data.lcquad_format_entry import LCQuadFormatEntry

class LCQuadRMDataLoader:

    def __init__(self, config, logger):
        self.conf = config
        self.logger = logger
        self.lcquad_tokenizer_obj = None
        self.tokenizer = None

    def customized_right_pad_collate_fn(
            self,
            batch
    ):

        # === 1. Extract entries ===
        org_txt = [
            LCQuadFormatEntry.prompt_rm_format_entry(item) for item in batch
        ]

        reward_score_lst = [
            float(item['reward_score']) for item in batch
        ]

        ignore_index = self.conf['model']['rm_model']['model_config']['ignore_index'] # -100
        max_len = self.conf['model']['rm_model']['model_config']['allowed_max_length']

        # tokenizer ID
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id
        sparql_token_id = self.tokenizer.convert_tokens_to_ids("<SPARQL_START>")

        # === 2. Tokenize entire batch at one go (much faster!) ===
        tok = self.lcquad_tokenizer_obj.lcquad_txt_encoder(org_txt, self.tokenizer)

        ip_token_ids = tok["input_ids"]  # list[list[int]]
        # === 3. Add EOS to each sequence ===
        for ids in ip_token_ids:
            ids.append(eos_token_id)

        # === 4. Determine padding length ===
        batch_max = max(len(ids) for ids in ip_token_ids)
        if max_len:
            batch_max = min(batch_max, max_len)

        # === 5. Build padded input_ids and labels ===
        ip_org_token_ids, ip_org_text_lst = [], []
        ip_padded_token_ids, ip_padded_text_lst = [], []

        for ids in ip_token_ids:

            ip_org_token_ids.append(ids.copy())
            ip_org_text = self.lcquad_tokenizer_obj.lcquad_tok_decoder(ids, self.tokenizer)

            ip_org_text_lst.append(ip_org_text)

            # TRUNCATE if needed
            if len(ids) > batch_max:
                ids = ids[:batch_max]

            # Pad input
            padded = ids + [pad_token_id] * (batch_max - len(ids))
            ip_padded_token_ids.append(padded.copy())
            ip_modf_text = self.lcquad_tokenizer_obj.lcquad_tok_decoder(padded, self.tokenizer)
            ip_padded_text_lst.append(ip_modf_text)

        # === 6. Convert to tensors ===
        device = self.conf['model']['device']
        ip_padded_token_ids = torch.tensor(ip_padded_token_ids, dtype=torch.long, device=device)
        reward_scores = torch.tensor(reward_score_lst, dtype=torch.float32, device=device)

        data_batch = {
            "prompt": org_txt,
            "ip_org_token_ids": ip_org_token_ids,
            "ip_org_text_lst": ip_org_text_lst,
            "ip_padded_token_ids": ip_padded_token_ids,
            "ip_padded_text_lst": ip_padded_text_lst,
            "reward_scores": reward_scores,
        }

        return data_batch

    def load_rm_dataloader(self, tokenizer, dataset, dataset_ind, padding_ind="right"):

        self.logger.info(f"generating dataloader for dataset {dataset_ind}, padding {padding_ind}")

        self.tokenizer = tokenizer
        self.lcquad_tokenizer_obj = LCQUADTokenizer(self.conf, self.logger)

        num_workers = self.conf['model']['rm_model']['num_workers']
        if dataset_ind == "train":
            batch_size = self.conf['model']['rm_model']['model_config']['batch_size']['train_batch_size']
        elif dataset_ind == "val":
            batch_size = self.conf['model']['rm_model']['model_config']['batch_size']['val_batch_size']
        elif dataset_ind == "test":
            batch_size = self.conf['model']['rm_model']['model_config']['batch_size']['test_batch_size']
        else:
            raise NotImplementedError

        if padding_ind == "right":
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=self.customized_right_pad_collate_fn,
                shuffle=False,
                num_workers=num_workers
            )
        elif padding_ind == "left":
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=self.customized_right_pad_collate_fn,
                shuffle=False,
                num_workers=num_workers
            )
        else:
            raise NotImplementedError

        return dataloader