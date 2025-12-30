from lcquad_finetuning.util.util_lib import *

class LCQuadRLHFDataLoader:

    def __init__(self, config, logger):
        self.conf = config
        self.logger = logger

    def customized_collate_fn(
            self,
            batch
    ):
        return {
            "prompts": [item["prompt"] for item in batch],
            "responses": [item["response"] for item in batch],
            "rewards": torch.tensor(
                [item["reward"] for item in batch],
                dtype=torch.float32
            ),
        }

    def load_rlhf_dataloader(self, dataset):

        batch_size = self.conf['model']['batch_size']['train_batch_size']
        num_workers = self.conf['model']['num_workers']

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.customized_collate_fn,
            shuffle=True,
            num_workers=num_workers
        )

        return dataloader