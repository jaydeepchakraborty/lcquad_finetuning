from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.data.lcquad_sft_dataset import LCQUADSFTDataset
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer
from lcquad_finetuning.data.lcquad_format_entry import LCQuadFormatEntry

class LCQuadINFDataLoader:

    def __init__(self, config, logger):
        self.conf = config
        self.logger = logger
        self.lcquad_tokenizer_obj = None
        self.tokenizer = None

    def customized_left_pad_collate_fn(self, batch):

        # === 1. Extract text ===
        prompts = [
           item['prompt_without_response'] for item in batch
        ]

        questionset_lst = [
            f"{item['question']}" for item in batch
        ]
        org_aparql_lst = [
            f"{item['org_sparql']}" for item in batch
        ]

        max_len = self.conf['model']['inf_model']['model_config']['allowed_max_length']

        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id

        # === 2. Tokenize ===
        tok = self.lcquad_tokenizer_obj.lcquad_txt_encoder(prompts, self.tokenizer)
        ip_token_ids = tok["input_ids"]  # list[list[int]]

        # === 3. Append EOS ===
        for ids in ip_token_ids:
            ids.append(eos_token_id)

        # === 4. Determine max length ===
        batch_max = max(len(ids) for ids in ip_token_ids)
        if max_len:
            batch_max = min(batch_max, max_len)

        # === 5. LEFT padding ===
        ip_modf_token_ids = []

        for ids in ip_token_ids:

            # truncate from LEFT if needed (keep most recent tokens)
            if len(ids) > batch_max:
                ids = ids[-batch_max:]

            pad_len = batch_max - len(ids)
            padded = [pad_token_id] * pad_len + ids  # LEFT PAD
            ip_modf_token_ids.append(padded)

        # === 6. Tensor ===
        device = self.conf['model']['device']
        ip_modf_token_ids = torch.tensor(
            ip_modf_token_ids,
            dtype=torch.long,
            device=device
        )

        return {
            "question": questionset_lst,
            "original_sparql": org_aparql_lst,
            "prompt_without_response": prompts,
            "ip_modf_token_ids": ip_modf_token_ids
        }

    def load_inf_dataloader(self, tokenizer, dataset, dataset_ind, padding_ind):

        self.logger.info(f"generating dataloader for dataset {dataset_ind}, padding {padding_ind}")

        self.tokenizer = tokenizer
        self.lcquad_tokenizer_obj = LCQUADTokenizer(self.conf, self.logger)

        num_workers = self.conf['model']['inf_model']['num_workers']
        batch_size = self.conf['model']['inf_model']['model_config']['batch_size']['test_batch_size']

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.customized_left_pad_collate_fn,
            shuffle=False,
            num_workers=num_workers
        )

        return dataloader