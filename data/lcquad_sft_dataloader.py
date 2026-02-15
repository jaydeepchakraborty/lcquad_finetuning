from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.data.lcquad_sft_dataset import LCQUADSFTDataset
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer
from lcquad_finetuning.data.lcquad_format_entry import LCQuadFormatEntry

class BucketBatchSampler(Sampler):
    """
    Groups samples by token length into buckets, shuffles bucket order
    and within-bucket order each epoch, then yields batches.
    Reduces padding waste while maintaining training randomness.
    """

    def __init__(self, token_lengths, batch_size, bucket_size_multiplier=10):
        self.batch_size = batch_size
        # Sort indices by token length
        sorted_indices = sorted(range(len(token_lengths)), key=lambda i: token_lengths[i])
        # Divide into buckets (each bucket has ~bucket_size_multiplier batches worth of samples)
        bucket_size = batch_size * bucket_size_multiplier
        self.buckets = [sorted_indices[i:i + bucket_size]
                        for i in range(0, len(sorted_indices), bucket_size)]

    def __iter__(self):
        # Shuffle bucket order each epoch
        bucket_order = list(range(len(self.buckets)))
        random.shuffle(bucket_order)

        for bi in bucket_order:
            bucket = list(self.buckets[bi])
            # Shuffle within bucket (preserves approximate length grouping)
            random.shuffle(bucket)
            # Yield batches from this bucket
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i:i + self.batch_size]

    def __len__(self):
        return sum(-(-len(b) // self.batch_size) for b in self.buckets)


class LCQuadSFTDataLoader:

    def __init__(self, config, logger):
        self.conf = config
        self.logger = logger
        self.lcquad_tokenizer_obj = None
        self.tokenizer = None
        self.prompt_col_nm = ""

    def customized_train_right_pad_collate_fn(
            self,
            batch
    ):

        # === 1. Extract entries ===
        org_txt = [item[self.prompt_col_nm] for item in batch]

        ignore_index = self.conf['model']['sft_model']['model_config']['ignore_index'] # -100
        max_len = self.conf['model']['sft_model']['model_config']['allowed_max_length']

        # tokenizer ID
        pad_token_id = self.tokenizer.pad_token_id
        # eos_token_id = self.tokenizer.eos_token_id
        sparql_token_id = self.tokenizer.convert_tokens_to_ids("<SPARQL_START>")

        # === 2. Tokenize entire batch at one go (much faster!) ===
        tok = self.lcquad_tokenizer_obj.lcquad_txt_encoder(org_txt, self.tokenizer)

        ip_token_ids = tok["input_ids"]  # list[list[int]]

        # === 4. Determine padding length ===
        batch_max = max(len(ids) for ids in ip_token_ids)
        if max_len:
            batch_max = min(batch_max, max_len)

        # === 5. Build padded input_ids and labels ===
        ip_org_token_ids, ip_org_text_lst = [], []
        ip_modf_token_ids, attn_mask_lst, ip_modf_text_lst = [], [], []

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

            attention_mask = [1 if t != pad_token_id else 0 for t in padded]
            attn_mask_lst.append(attention_mask)

            ip_modf_text = self.lcquad_tokenizer_obj.lcquad_tok_decoder(padded, self.tokenizer)
            ip_modf_text_lst.append(ip_modf_text)

            # ==== BUILD TARGET LABELS (aligned with input_ids, no manual shift) ====
            # HuggingFace model(input_ids, labels=labels).loss shifts internally
            labels = padded.copy()
            lbl_org_token_ids.append(labels.copy())
            lbl_org_text = self.lcquad_tokenizer_obj.lcquad_tok_decoder(labels, self.tokenizer)
            lbl_org_text_lst.append(lbl_org_text)
            """
            input_ids:      <Q_START>  question  <Q_END>  <SPARQL_START>  SELECT  ?answer  <SPARQL_END>  PAD
            attention_mask:  1          1         1        1               1       1        1             0
            labels:         -100       -100      -100     -100            SELECT  ?answer  <SPARQL_END>  -100
            """
            # MASK EVERYTHING BEFORE (and including) <SPARQL_START>
            try:
                idx = padded.index(sparql_token_id)
            except ValueError:
                idx = -1
            if idx != -1:
                for j in range(idx + 1):
                    labels[j] = ignore_index
            else:
                # <SPARQL_START> not found (truncated) — skip this sample's loss
                labels = [ignore_index] * len(labels)
            # MASK PAD
            labels = [ignore_index if t == pad_token_id else t for t in labels]
            lbl_modf_text = self.lcquad_tokenizer_obj.lcquad_tok_decoder(labels, self.tokenizer)
            lbl_modf_text_lst.append(lbl_modf_text)
            lbl_modf_token_ids.append(labels)

        # === 6. Convert to tensors ===
        device = self.conf['model']['device']
        ip_modf_token_ids = torch.tensor(ip_modf_token_ids,
                                         dtype=torch.long,
                                         device=device)
        attention_mask = torch.tensor(attn_mask_lst,
                                        dtype=torch.long,
                                        device=device
                                    )
        lbl_modf_token_ids = torch.tensor(lbl_modf_token_ids,
                                          dtype=torch.long,
                                          device=device)

        data_batch = {
            "org_txt": org_txt,
            "ip_org_token_ids": ip_org_token_ids,
            "ip_org_text_lst": ip_org_text_lst,
            "ip_modf_token_ids": ip_modf_token_ids,
            "ip_modf_text_lst": ip_modf_text_lst,
            "attention_mask": attention_mask,
            "lbl_org_token_ids": lbl_org_token_ids,
            "lbl_org_text_lst": lbl_org_text_lst,
            "lbl_modf_token_ids": lbl_modf_token_ids,
            "lbl_modf_text_lst": lbl_modf_text_lst,
        }

        return data_batch

    def customized_test_right_pad_collate_fn(self, batch):

        # === 1. Extract text ===
        org_txt_lst = [item[self.prompt_col_nm] for item in batch]

        questionset_lst = [item["question"] for item in batch]
        org_sparql_lst = [item["sparql"] for item in batch]

        max_len = self.conf['model']['sft_model']['model_config']['allowed_max_length']

        pad_token_id = self.tokenizer.pad_token_id

        # === 2. Tokenize ===
        tok = self.lcquad_tokenizer_obj.lcquad_txt_encoder(
            org_txt_lst, self.tokenizer
        )
        ip_token_ids = tok["input_ids"]  # list[list[int]]

        # === 3. Determine max length ===
        batch_max = max(len(ids) for ids in ip_token_ids)
        if max_len:
            batch_max = min(batch_max, max_len)

        # === 5. RIGHT padding + attention mask ===
        ip_modf_token_id_lst = []
        attention_mask_lst = []

        for ids in ip_token_ids:

            # truncate from RIGHT (keep prefix)
            if len(ids) > batch_max:
                ids = ids[:batch_max]

            pad_len = batch_max - len(ids)

            padded_ids = ids + [pad_token_id] * pad_len  # RIGHT PAD
            attn_mask = [1] * len(ids) + [0] * pad_len  # RIGHT MASK

            ip_modf_token_id_lst.append(padded_ids)
            attention_mask_lst.append(attn_mask)

        # === 6. Tensor ===
        device = self.conf['model']['device']

        ip_modf_token_ids_tensor = torch.tensor(
            ip_modf_token_id_lst,
            dtype=torch.long,
            device=device
        )

        attention_mask_tensor = torch.tensor(
            attention_mask_lst,
            dtype=torch.long,
            device=device
        )

        return {
            "org_txt": org_txt_lst,
            "question": questionset_lst,
            "org_sparql": org_sparql_lst,
            "ip_modf_token_ids": ip_modf_token_ids_tensor,
            "attention_mask": attention_mask_tensor,
        }

    def customized_left_pad_collate_fn(self, batch):

        # === 1. Extract text ===
        org_txt_lst = [item[self.prompt_col_nm] for item in batch]

        questionset_lst = [
            f"{item['question']}" for item in batch
        ]
        org_aparql_lst = [
            f"{item['sparql']}" for item in batch
        ]

        max_len = self.conf['model']['sft_model']['model_config']['allowed_max_length']

        pad_token_id = self.tokenizer.pad_token_id

        # === 2. Tokenize ===
        tok = self.lcquad_tokenizer_obj.lcquad_txt_encoder(org_txt_lst, self.tokenizer)
        ip_token_ids = tok["input_ids"]  # list[list[int]]

        # === 3. Determine max length ===
        batch_max = max(len(ids) for ids in ip_token_ids)
        if max_len:
            batch_max = min(batch_max, max_len)

        # === 4. LEFT padding ===
        ip_modf_token_id_lst = []
        attention_mask_lst = []

        for ids in ip_token_ids:

            # truncate from LEFT if needed (keep most recent tokens)
            if len(ids) > batch_max:
                ids = ids[-batch_max:]

            pad_len = batch_max - len(ids)

            padded_ids = [pad_token_id] * pad_len + ids  # LEFT PAD
            attn_mask = [0] * pad_len + [1] * len(ids)

            ip_modf_token_id_lst.append(padded_ids)
            attention_mask_lst.append(attn_mask)

        # === 6. Tensor ===
        device = self.conf['model']['device']
        ip_modf_token_id_lst_tensor = torch.tensor(
            ip_modf_token_id_lst,
            dtype=torch.long,
            device=device
        )
        attention_mask_tensor = torch.tensor(
            attention_mask_lst,
            dtype=torch.long,
            device=device
        )

        return {
            "org_txt": org_txt_lst,
            "question": questionset_lst,
            "org_sparql": org_aparql_lst,
            "ip_modf_token_ids": ip_modf_token_id_lst_tensor,
            "attention_mask": attention_mask_tensor
        }

    def load_sft_dataloader(self, tokenizer, dataset, dataset_ind, padding_ind, col_ind):

        self.logger.info(f"generating dataloader for dataset {dataset_ind}, padding {padding_ind}, col_ind {col_ind}")

        self.tokenizer = tokenizer
        self.lcquad_tokenizer_obj = LCQUADTokenizer(self.conf, self.logger)

        num_workers = self.conf['model']['sft_model']['num_workers']
        if dataset_ind == "train":
            batch_size = self.conf['model']['sft_model']['model_config']['batch_size']['train_batch_size']
        elif dataset_ind == "val":
            batch_size = self.conf['model']['sft_model']['model_config']['batch_size']['val_batch_size']
        elif dataset_ind == "test":
            batch_size = self.conf['model']['sft_model']['model_config']['batch_size']['test_batch_size']
        else:
            raise NotImplementedError

        self.prompt_col_nm = col_ind

        if padding_ind == "right" and dataset_ind == "train":
            # Compute token lengths for bucket sampling
            all_texts = [dataset[i][col_ind] for i in range(len(dataset))]
            tok = self.lcquad_tokenizer_obj.lcquad_txt_encoder(all_texts, self.tokenizer)
            token_lengths = [len(ids) for ids in tok["input_ids"]]

            bucket_sampler = BucketBatchSampler(token_lengths, batch_size)
            dataloader = DataLoader(
                dataset,
                batch_sampler=bucket_sampler,
                collate_fn=self.customized_train_right_pad_collate_fn,
                num_workers=num_workers
            )
        elif padding_ind == "right" and dataset_ind == "val":
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=self.customized_train_right_pad_collate_fn,
                shuffle=False,
                num_workers=num_workers
            )
        elif padding_ind == "right" and dataset_ind == "test":
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=self.customized_test_right_pad_collate_fn,
                shuffle=False,
                num_workers=num_workers
            )
        elif padding_ind == "left" and dataset_ind == "test":
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=self.customized_left_pad_collate_fn,
                shuffle=False,
                num_workers=num_workers
            )
        else:
            raise NotImplementedError

        return dataloader