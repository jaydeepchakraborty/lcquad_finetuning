from lcquad_finetuning.util.util_lib import *


class OfflinePPOTrainer:
    def __init__(self, policy_model, ref_model, dataloader, clip_ratio=0.2, lr=5e-5, device="cpu"):
        """
        Offline PPO Trainer

        Args:
            policy_model: LoRA-trained SFT model to update
            ref_model: frozen reference model (no updates)
            dataloader: provides batches with keys:
                        "logprob_old", "reward_score", "value_score"
            clip_ratio: PPO clipping epsilon
            lr: optimizer learning rate
        """
        self.policy_model = policy_model.to(device)
        self.ref_model = ref_model.to(device)
        self.ref_model.eval()  # ensure frozen
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.dataloader = dataloader
        self.clip_ratio = clip_ratio
        self.device = device

        # Only LoRA parameters should be updated
        self.optimizer = torch.optim.AdamW(
            [p for p in self.policy_model.parameters() if p.requires_grad],
            lr=lr
        )

    def compute_loss(self, logprob_new, logprob_old, advantage):
        """
        PPO clipped objective
        """
        ratio = torch.exp(logprob_new - logprob_old)  # [B]
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
        return loss

    def train(self, epochs=1):
        self.policy_model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in self.dataloader:
                # Move to device
                logprob_old = batch["logprob_old"].to(self.device)  # [B]
                reward_score = batch["reward_score"].to(self.device)  # [B]
                value_score = batch["value_score"].to(self.device)  # [B]

                # Compute advantage
                advantage = reward_score - value_score  # [B]

                # Compute logprob of policy_model on same data
                # Assuming you already have tokenized input in batch
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.policy_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                # Get logprobs for generated sequence
                # For causal LM: sum log softmax over sequence tokens
                logits = outputs.logits  # [B, T, V]
                target_ids = batch["generated_ids"].to(self.device)  # [B, T]

                log_probs = F.log_softmax(logits, dim=-1)
                logprob_new = torch.gather(log_probs, -1, target_ids.unsqueeze(-1)).squeeze(-1).sum(dim=-1)  # [B]

                # PPO loss
                loss = self.compute_loss(logprob_new, logprob_old, advantage)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch + 1} | PPO loss: {avg_loss:.4f}")
