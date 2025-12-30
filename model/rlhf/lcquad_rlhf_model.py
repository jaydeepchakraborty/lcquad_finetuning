from lcquad_finetuning.util.lcquad_util import LCQuadUtil
from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer
from lcquad_finetuning.util.lcquad_exception import LCQUADException

class LCQUADRLHFModel:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def train_lcquad_rlhf_model(self, policy_model, policy_tokenizer, train_loader):

        # PPO configuration
        ppo_config = PPOConfig(
            learning_rate=5e-5,
            mini_batch_size=1,
            gradient_accumulation_steps=1
        )

        ref_model = copy.deepcopy(policy_model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        # Load your SFT policy model
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=policy_model,
            ref_model=ref_model,
            tokenizer=policy_tokenizer
        )

        # Iterate over your dataset with precomputed reward_score
        for batch in train_loader:
            prompts = batch["prompt"]
            responses = batch["response"]
            rewards = batch["reward"]

            ppo_trainer.step(
                prompts,
                responses,
                rewards
            )

        return ppo_trainer

    def save_lcquad_rlhf_model(self, lcquad_rlhf_model):

        rlhf_model_path = self.config['model']['rlhf_model_path']
        rlhf_model_path = rlhf_model_path.replace("latest", LCQuadUtil.get_curr_tm())

        if self.config['model']['chosen_model'] == "gpt2":
            lcquad_rlhf_model.save_pretrained(rlhf_model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            lcquad_rlhf_model.save_pretrained(rlhf_model_path)
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        self.logger.info(f"model saved to {rlhf_model_path}")

        sft_model_path = self.config['model']['sft_model_path']

        if self.config['model']['chosen_model'] == "gpt2":
            lcquad_rlhf_model.save_pretrained(sft_model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            lcquad_rlhf_model.save_pretrained(sft_model_path)
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        self.logger.info(f"model saved to {sft_model_path}")

