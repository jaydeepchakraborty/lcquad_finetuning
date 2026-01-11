from lcquad_finetuning.data.lcquad_datahelper import LCQUADDataHelper
from lcquad_finetuning.model.rm.lcquad_rm_model import LCQUADRMModel
from lcquad_finetuning.model.sft.lcquad_sft_modelhelper import LCQUADSFTMODELHelper
from lcquad_finetuning.model.vm.lcquad_vm_model import LCQUADVMModel
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer
from lcquad_finetuning.util.lcquad_exception import LCQUADException
from lcquad_finetuning.model.rm.lcquad_rm_reward_score import LCQUADRMRewardScoreGenerator
from lcquad_finetuning.util.lcquad_util import LCQuadUtil
from lcquad_finetuning.util.util_lib import *

class LCQUADVMMODELHelper:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def load_tokenizer(self):
        lcquad_tokenizer_obj = LCQUADTokenizer(self.config, self.logger)
        tokenizer = lcquad_tokenizer_obj.load_tokenizer()
        return tokenizer

    def load_vm_dataloder(self):
        # loading the tokenizer
        tokenizer = self.load_tokenizer()

        lcquad_data_loader_obj = LCQUADDataHelper(self.config, self.logger)
        dataset_file_path = self.config['data']["rm_train_with_reward_score_dataset"]
        train_dataloader = lcquad_data_loader_obj.load_vm_dataloader(tokenizer, dataset_file_path, "train", "right")
        self.logger.info(f"train dataloader batches:- {len(train_dataloader)}")

        dataset_file_path = self.config['data']["rm_test_with_reward_score_dataset"]
        test_dataloader = lcquad_data_loader_obj.load_vm_dataloader(tokenizer, dataset_file_path, "test", "right")
        self.logger.info(f"test dataloader batches:- {len(test_dataloader)}")

        return train_dataloader, test_dataloader


    def load_lcquad_clm_model(self):

        model_path = self.config['model']['clm_model_path']
        self.logger.info(f"loading model from {model_path}")

        if self.config['model']['chosen_model'] == "gpt2":
            model_obj = GPT2LMHeadModel.from_pretrained(model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            model_obj = AutoModel.from_pretrained(model_path, dtype=torch.float32, device_map=None)
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        return model_obj

    def train_vm_model(self, vm_model, train_vm_dataloader, tokenizer):

        device = self.config['model']['device']
        vm_model.to(device)

        effective_batch_size = self.config['model']['batch_size']['effective_batch_size']
        real_batch_size = self.config['model']['batch_size']['train_batch_size']
        accum_steps = effective_batch_size // real_batch_size

        optimizer = torch.optim.AdamW(vm_model.value_head.parameters(), lr=1e-5)
        loss_fn = torch.nn.MSELoss()

        vm_model.train()
        vm_model.model.eval()  # freeze base LM

        num_epochs = self.config['model']['num_epochs']

        for epoch in range(num_epochs):
            total_loss = 0.0
            accum_counter = 0
            optimizer.zero_grad()

            for batch_data in train_vm_dataloader:
                input_ids = batch_data['ip_padded_token_ids'].to(device)
                rewards_gt = batch_data['reward_scores'].float().to(device)

                attention_mask = (input_ids != tokenizer.pad_token_id).long()

                outputs = vm_model(input_ids=input_ids, attention_mask=attention_mask)
                values = outputs.logits.squeeze(-1)  # [B, T]

                seq_lens = attention_mask.sum(dim=1) - 1
                rewards_pred = values[torch.arange(values.size(0)), seq_lens]

                raw_loss = loss_fn(rewards_pred, rewards_gt)
                loss = raw_loss / accum_steps
                loss.backward()

                accum_counter += 1
                total_loss += raw_loss.item()

                if accum_counter == accum_steps:
                    torch.nn.utils.clip_grad_norm_(vm_model.value_head.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_counter = 0

            if accum_counter > 0:
                torch.nn.utils.clip_grad_norm_(vm_model.value_head.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = total_loss / len(train_vm_dataloader)
            self.logger.info(f"Epoch {epoch + 1} | VM loss: {avg_loss:.4f}")

        return vm_model

    def train_value_model_helper(self):

        # loading the vm dataloader
        train_vm_dataloader, test_vm_dataloader = self.load_vm_dataloder()

        # loading tokenizer
        tokenizer = self.load_tokenizer()

        # loading the base Causal Language model (trained on new tokens)
        clm_model = self.load_lcquad_clm_model()

        device = self.config["model"]["device"]
        vm_model_obj = LCQUADVMModel(clm_model, self.config, self.logger).to(device)
        vm_model = self.train_vm_model(vm_model_obj, train_vm_dataloader, tokenizer)

        return vm_model

    def save_value_model(self, vm_model):
        # save reward model
        save_dir = self.config['model']['vm_model_path']
        vm_model.model.save_pretrained(save_dir)
        # save reward head
        torch.save(
            vm_model.head.state_dict(),
            f"{save_dir}/reward_head.pt"
        )
        self.logger.info(f"save reward model to {save_dir}")

        save_dir = save_dir.replace('latest', LCQuadUtil.get_curr_tm())
        vm_model.model.save_pretrained(save_dir)
        # save value head
        torch.save(
            vm_model.head.state_dict(),
            f"{save_dir}/value_head.pt"
        )
        self.logger.info(f"save value model to {save_dir}")

        return
