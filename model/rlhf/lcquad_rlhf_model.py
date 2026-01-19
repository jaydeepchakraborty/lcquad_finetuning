from lcquad_finetuning.util.lcquad_util import LCQuadUtil
from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer
from lcquad_finetuning.util.lcquad_exception import LCQUADException

class LCQUADRLHFModel:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def compute_loss(self, logprob_new, logprob_old, advantage):
        ratio = torch.exp(logprob_new - logprob_old)
        clip_ratio = 0.2
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
        return loss

    def train_lcquad_rlhf_model(self, policy_tokenizer,
                                      policy_train_dataloader,
                                      policy_model,
                                      reference_model,
                                      reward_model,
                                      value_model):

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [p for p in policy_model.parameters() if p.requires_grad],
                    "lr": 5e-5,  # policy LR
                },
                {
                    "params": [p for p in value_model.parameters() if p.requires_grad],
                    "lr": 5e-5,  # value LR (can be different)
                },
            ]
        )

        device = self.config['model']['device']
        max_new_tokens = self.config['model']['rlhf_model']['model_config']['allowed_tokens']

        policy_model.train()
        value_model.train()

        effective_batch_size = self.config['model']['rlhf_model']['model_config']['batch_size']['effective_batch_size']
        real_batch_size = self.config['model']['rlhf_model']['model_config']['batch_size']['train_batch_size']
        accum_steps = effective_batch_size // real_batch_size

        optimizer.zero_grad()
        accum_counter = 0

        epochs = self.config['model']['rlhf_model']['model_config']['num_epochs']
        for epoch in range(epochs):
            total_policy_loss = 0.0
            total_value_loss = 0.0

            for batch in policy_train_dataloader:

                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]

                # --- 1. Generate output from policy model ---
                generated_outputs = policy_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1, # how many outputs are generated
                    pad_token_id=policy_tokenizer.pad_token_id,
                    eos_token_id=policy_tokenizer.eos_token_id
                )
                generated_ids = generated_outputs[:, input_ids.shape[1]:]  # only new tokens
                generated_attention_mask =  torch.ones_like(generated_ids, device=device)

                # --- 2. Compute old log_probs with frozen reference model ---
                with torch.no_grad():
                    outputs_ref = reference_model(input_ids=torch.cat([input_ids, generated_ids], dim=1),
                                                 attention_mask=torch.cat([attention_mask, generated_attention_mask], dim=1),
                                                 return_dict=True)
                    logits_ref = outputs_ref.logits
                    log_probs_ref = F.log_softmax(logits_ref, dim=-1)
                    log_prob_old = torch.gather(log_probs_ref, -1, generated_ids.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

                # --- 3. Compute new log_probs with policy model ---
                outputs = policy_model(input_ids=torch.cat([input_ids, generated_ids], dim=1),
                                            attention_mask=torch.cat([attention_mask, generated_attention_mask], dim=1),
                                            return_dict=True)
                logits = outputs.logits
                log_probs = F.log_softmax(logits, dim=-1)
                log_prob_new = torch.gather(log_probs, -1, generated_ids.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

                # --- 4. Compute reward and value for generated output ---
                reward_score = reward_model(
                    input_ids=torch.cat([input_ids, generated_ids], dim=1),
                    attention_mask=torch.cat([attention_mask, generated_attention_mask], dim=1)
                )
                reward_score = reward_score.to(torch.float16)
                value_score = value_model(
                    input_ids=torch.cat([input_ids, generated_ids], dim=1),
                    attention_mask=torch.cat([attention_mask, generated_attention_mask], dim=1)
                )
                value_score = value_score.mean(dim=1)  # [B, T] -> [B]
                value_score = value_score.to(torch.float16)

                advantage = reward_score - value_score

                # --- 5. loss and update ---
                # Policy loss (PPO)
                policy_loss = self.compute_loss(log_prob_new, log_prob_old, advantage)
                # Value loss (MSE)
                value_loss = F.mse_loss(value_score, reward_score)

                total_loss = policy_loss + value_loss

                loss = total_loss / accum_steps
                loss.backward()

                accum_counter += 1
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

                # ---- optimizer step every accum_steps ----
                if accum_counter == accum_steps:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in policy_model.parameters() if p.requires_grad] +
                        [p for p in value_model.parameters() if p.requires_grad],
                        1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_counter = 0

            # flush leftovers
            if accum_counter > 0:
                optimizer.step()
                optimizer.zero_grad()

            self.logger.info(
                f"Epoch {epoch + 1} | "
                f"Policy: {total_policy_loss / len(policy_train_dataloader):.4f} | "
                f"Value: {total_value_loss / len(policy_train_dataloader):.4f}"
            )

        return policy_model