from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.data.lcquad_sft_dataset import LCQUADSFTDataset
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer

class LCQuadSFTDataLoader:

    def __init__(self, config, logger):
        self.conf = config
        self.logger = logger
        self.lcquad_tokenizer_obj = None
        self.tokenizer = None

    def customized_collate_fn(
            self,
            batch
    ):

        # === 1. Extract entries ===
        org_txt = [item["entry"] for item in batch]

        ignore_index = self.conf['model']['gpt_config']['basic_config']['ignore_index']
        max_len = self.conf['model']['gpt_config']['basic_config']['allowed_max_length']

        # tokenizer ID
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id
        sparql_token_id = self.tokenizer.convert_tokens_to_ids("<SPARQL>")

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
        ip_modf_token_ids, ip_modf_text_lst = [], []

        lbl_org_token_ids, lbl_org_text_lst = [], []
        lbl_modf_token_ids, lbl_modf_text_lst = [], []

        for ids in ip_token_ids:

            ip_org_token_ids.append(ids.copy())
            ip_org_text = self.lcquad_tokenizer_obj.lcquad_tok_decoder(ids, self.tokenizer)

            ip_org_text_lst.append(ip_org_text)

            # TRUNCATE if needed
            if len(ids) > batch_max:
                ids = ids[:batch_max]

            # Pad input
            padded = ids + [pad_token_id] * (batch_max - len(ids))
            ip_modf_token_ids.append(padded.copy())
            ip_modf_text = self.lcquad_tokenizer_obj.lcquad_tok_decoder(padded, self.tokenizer)
            ip_modf_text_lst.append(ip_modf_text)

            # ==== BUILD TARGET LABELS ====
            labels = padded.copy()
            # SHIFT LEFT
            for i in range(len(labels) - 1):
                labels[i] = labels[i + 1]
            labels[-1] = pad_token_id
            lbl_org_token_ids.append(labels.copy())
            lbl_org_text = self.lcquad_tokenizer_obj.lcquad_tok_decoder(labels, self.tokenizer)
            lbl_org_text_lst.append(lbl_org_text)
            # MASK EVERYTHING BEFORE (and including) <SPARQL>
            try:
                idx = padded.index(sparql_token_id)
            except ValueError:
                idx = -1
            if idx != -1:
                for j in range(idx + 1):
                    labels[j] = ignore_index
            # MASK PAD
            labels = [ignore_index if t == pad_token_id else t for t in labels]
            lbl_modf_text = self.lcquad_tokenizer_obj.lcquad_tok_decoder(labels, self.tokenizer)
            lbl_modf_text_lst.append(lbl_modf_text)
            lbl_modf_token_ids.append(labels)

        # === 6. Convert to tensors ===
        device = self.conf['model']['device']
        ip_modf_token_ids = torch.tensor(ip_modf_token_ids, dtype=torch.long, device=device)
        lbl_modf_token_ids = torch.tensor(lbl_modf_token_ids, dtype=torch.long, device=device)

        data_batch = {
            "org_txt": org_txt,
            "ip_org_token_ids": ip_org_token_ids,
            "ip_org_text_lst": ip_org_text_lst,
            "ip_modf_token_ids": ip_modf_token_ids,
            "ip_modf_text_lst": ip_modf_text_lst,
            "lbl_org_token_ids": lbl_org_token_ids,
            "lbl_org_text_lst": lbl_org_text_lst,
            "lbl_modf_token_ids": lbl_modf_token_ids,
            "lbl_modf_text_lst": lbl_modf_text_lst,
        }

        return data_batch

    def load_sft_dataloader(self, tokenizer, dataset, dataset_ind):

        self.tokenizer = tokenizer
        self.lcquad_tokenizer_obj = LCQUADTokenizer(self.conf, self.logger)

        num_workers = self.conf['model']['num_workers']
        if dataset_ind == "train":
            batch_size = self.conf['model']['batch_size']['train_batch_size']
        elif dataset_ind == "val":
            batch_size = self.conf['model']['batch_size']['val_batch_size']
        elif dataset_ind == "test":
            batch_size = self.conf['model']['batch_size']['test_batch_size']
        else:
            raise NotImplementedError

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.customized_collate_fn,
            shuffle=False,
            num_workers=num_workers
        )

        return dataloader