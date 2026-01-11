from lcquad_finetuning.util.lcquad_util import LCQuadUtil
from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer
from lcquad_finetuning.util.lcquad_exception import LCQUADException

class LCQUADRLHFModel:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def train_lcquad_rlhf_model(self, policy_tokenizer, policy_model, ref_model, reward_model, value_model, train_dataset):

        # policy_model.stop_token = policy_tokenizer.eos_token
        # policy_model.stop_token_id = policy_tokenizer.eos_token_id

        # PPO configuration
        ppo_config = PPOConfig(
            learning_rate=5e-5,
            mini_batch_size=1,
            gradient_accumulation_steps=1
        )

        generation_kwargs = {
            "max_new_tokens": 64,
            "eos_token_id": policy_tokenizer.eos_token_id,
            "pad_token_id": policy_tokenizer.pad_token_id,
            "do_sample": False,
        }

        ppo_trainer = PPOTrainer(
            policy_model,
            ref_model,
            policy_tokenizer,
            ppo_config,
            reward_model=reward_model,
            train_dataset=train_dataset,
            value_model=value_model
        )

        # Iterate over your dataset with precomputed reward_score
        for batch in train_dataset:
            prompts = [batch["prompt"]]
            # responses = [batch["response"]]
            # rewards = torch.tensor([batch["reward"]], dtype=torch.float32)

            ppo_trainer.step(
                prompts=prompts,
                # responses,
                # rewards,
                generation_kwargs=generation_kwargs
            )

        return ppo_trainer

