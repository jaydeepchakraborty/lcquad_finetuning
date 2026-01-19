from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer
from lcquad_finetuning.data.lcquad_format_entry import LCQuadFormatEntry

class LCQuadRLHFDataLoader:

    def __init__(self, config, logger):
        self.conf = config
        self.logger = logger
        self.lcquad_tokenizer_obj = None
        self.tokenizer = None

    def customized_left_pad_collate_fn(self, batch):

        # 1. load prompts
        rlhf_texts = [
            item["prompt_without_response"] for item in batch
        ]

        # 2. Tokenize batch (no padding here)
        tok = self.lcquad_tokenizer_obj.lcquad_txt_encoder(
            rlhf_texts,
            self.tokenizer
        )

        input_ids = tok["input_ids"]  # list[list[int]]
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id

        # 3. Append EOS
        for ids in input_ids:
            ids.append(eos_token_id)

        # 4. Truncation
        max_len = self.conf["model"]["rlhf_model"]["model_config"]["allowed_max_length"]
        batch_max = min(max(len(ids) for ids in input_ids), max_len)

        padded_ids, attention_masks = [], []

        for ids in input_ids:
            # truncate from the LEFT if too long (important for causal models)
            ids = ids[-batch_max:]
            pad_len = batch_max - len(ids)

            # LEFT padding
            padded = [pad_token_id] * pad_len + ids
            mask = [0] * pad_len + [1] * len(ids)

            padded_ids.append(padded)
            attention_masks.append(mask)

        # 5. Convert to tensors
        device = self.conf["model"]["device"]

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long, device=device),
            "prompt_without_response": rlhf_texts,  # optional
        }

    def customized_right_pad_collate_fn(self, batch):

        # 1. load prompts
        rlhf_texts = [
            item["prompt_without_response"] for item in batch
        ]

        # 2. Tokenize batch
        tok = self.lcquad_tokenizer_obj.lcquad_txt_encoder(
            rlhf_texts,
            self.tokenizer
        )

        input_ids = tok["input_ids"]  # list[list[int]]
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id

        # 3. Append EOS
        for ids in input_ids:
            ids.append(eos_token_id)

        # 4. Padding & truncation
        max_len = self.conf["model"]['rlhf_model']["model_config"]["allowed_max_length"]
        batch_max = min(max(len(ids) for ids in input_ids), max_len)

        padded_ids, attention_masks = [], []

        for ids in input_ids:
            ids = ids[:batch_max]
            pad_len = batch_max - len(ids)

            padded = ids + [pad_token_id] * pad_len
            mask = [1] * len(ids) + [0] * pad_len

            padded_ids.append(padded)
            attention_masks.append(mask)

        # 5. Convert to tensors
        device = self.conf["model"]["device"]

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long, device=device),
            "prompt_without_response": rlhf_texts,  # optional (for debugging)
        }

    def load_rlhf_dataloader(self, tokenizer, dataset, dataset_ind, padding_ind):

        self.logger.info(f"generating dataloader for dataset {dataset_ind}, padding {padding_ind}")

        self.tokenizer = tokenizer
        self.lcquad_tokenizer_obj = LCQUADTokenizer(self.conf, self.logger)

        num_workers = self.conf['model']['rlhf_model']['num_workers']
        if dataset_ind == "train":
            batch_size = self.conf['model']['rlhf_model']['model_config']['batch_size']['train_batch_size']
        elif dataset_ind == "val":
            batch_size = self.conf['model']['rlhf_model']['model_config']['batch_size']['val_batch_size']
        elif dataset_ind == "test":
            batch_size = self.conf['model']['rlhf_model']['model_config']['batch_size']['test_batch_size']
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
            # default ~ right padding
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