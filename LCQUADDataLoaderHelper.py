from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.LCQUADDataset import LCQUADDataset

class LCQUADDataLoaderHelper:

    def __init__(self, config: dict):
        self.conf = config

    def customized_collate_fn(
            self,
            batch,
            device="cpu"
    ):

        pad_token_id = self.conf['model']['gpt_config']['basic_config']['pad_token_id']
        ignore_index = self.conf['model']['gpt_config']['basic_config']['ignore_index']
        allowed_max_length = self.conf['model']['gpt_config']['basic_config']['allowed_max_length']

        # Find the longest sequence in the batch
        batch_max_length = max(len(item['encoded_text']) + 1 for item in batch)

        # Pad and prepare inputs and targets
        org_txt = []
        inputs_dec_txt, targets_dec_txt = [], []
        inputs_enc_lst, targets_enc_lst = [], []

        for item in batch:
            encoded_item = item['encoded_text'].copy()
            # Add an <|endoftext|> token
            encoded_item += [pad_token_id]
            # Pad sequences to max_length
            encoded_padded = (
                    encoded_item + [pad_token_id] *
                    (batch_max_length - len(encoded_item))
            )

            encoded_inputs = torch.tensor(encoded_padded[:-1])  # Truncate the last token for inputs
            encoded_targets = torch.tensor(encoded_padded[1:])  # Shift +1 to the right for targets

            # New: Replace all but the first padding tokens in targets by ignore_index
            mask = encoded_targets == pad_token_id
            indices = torch.nonzero(mask).squeeze()
            if indices.numel() > 1:
                encoded_targets[indices[1:]] = ignore_index

            # New: Optionally truncate to maximum sequence length
            if allowed_max_length is not None:
                encoded_inputs = encoded_inputs[:allowed_max_length]
                encoded_targets = encoded_targets[:allowed_max_length]

            inputs_enc_lst.append(encoded_inputs)
            targets_enc_lst.append(encoded_targets)

            org_item = item['decoded_text'].copy()

            # Add an <|endoftext|> token
            org_item += ["<|endoftext|>"]
            # Pad sequences to max_length
            org_padded = (
                    org_item + ["<|endoftext|>"] *
                    (batch_max_length - len(org_item))
            )

            inputs_dec_txt.append(org_padded)
            targets_dec_txt.append(org_padded[1:])

            org_txt.append(item['org_text'])

        # Convert list of inputs and targets to tensors and transfer to target device
        inputs_tensor = torch.stack(inputs_enc_lst).to(device)
        targets_tensor = torch.stack(targets_enc_lst).to(device)

        data_batch = {
            "org_txt": org_txt,
            "inputs_txt": inputs_dec_txt,
            "targets_txt": targets_dec_txt,
            "inputs_tensor": inputs_tensor,
            "targets_tensor": targets_tensor,
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