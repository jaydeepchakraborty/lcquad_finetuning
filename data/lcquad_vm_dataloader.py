from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer
from lcquad_finetuning.data.lcquad_format_entry import LCQuadFormatEntry

class LCQuadVMDataLoader:

    def __init__(self, config, logger):
        self.conf = config
        self.logger = logger
        self.lcquad_tokenizer_obj = None
        self.tokenizer = None

    def customized_right_pad_collate_fn(self, batch):

        # 1. Build VM input text
        vm_texts = [
            LCQuadFormatEntry.rm_format_entry(item, split="train")
            for item in batch
        ]

        rewards = torch.tensor(
            [item["reward_score"] for item in batch],
            dtype=torch.float32,
            device=self.conf["model"]["device"]
        )

        # 2. Tokenize batch
        tok = self.lcquad_tokenizer_obj.lcquad_txt_encoder(
            vm_texts,
            self.tokenizer
        )

        input_ids = tok["input_ids"]  # list[list[int]]
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id

        # 3. Append EOS
        for ids in input_ids:
            ids.append(eos_token_id)

        # 4. Padding & truncation
        max_len = self.conf["model"]["model_config"]["basic_config"]["allowed_max_length"]
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
            "reward_scores": rewards,
            "texts": vm_texts,  # optional (for debugging)
        }

    def load_vm_dataloader(self, tokenizer, dataset, dataset_ind, padding_ind="right"):

        self.logger.info(f"generating dataloader for dataset {dataset_ind}, padding {padding_ind}")

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
            # default
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=self.customized_right_pad_collate_fn,
                shuffle=False,
                num_workers=num_workers
            )

        return dataloader