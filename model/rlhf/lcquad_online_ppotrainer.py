from lcquad_finetuning.util.util_lib import *


class OnlinePPOTrainer:
    def __init__(self, policy_model, ref_model, reward_model, value_model, dataloader, clip_ratio=0.2, lr=5e-5,
                 device="cpu", max_new_tokens=64):
        """
        Online PPO Trainer

        Args:
            policy_model: LoRA-trained SFT model to update
            ref_model: frozen reference model
            reward_model: pretrained reward model
            value_model: pretrained value model
            dataloader: DataLoader with 'prompt' (text)
            clip_ratio: PPO clipping epsilon
            lr: learning rate
            device: "cuda"/"cpu"/"mps"
            max_new_tokens: generation length
        """
        self.policy_model = policy_model.to(device)
        self.ref_model = ref_model.to(device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.reward_model = reward_model.to(device)
        self.reward_model.eval()
        for p in self.reward_model.parameters():
            p.requires_grad = False

        self.value_model = value_model.to(device)
        self.value_model.eval()
        for p in self.value_model.parameters():
            p.requires_grad = False

        self.dataloader = dataloader
        self.clip_ratio = clip_ratio
        self.device = device
        self.max_new_tokens = max_new_tokens

        # Only LoRA parameters updated
        self.optimizer = torch.optim.AdamW(
            [p for p in self.policy_model.parameters() if p.requires_grad],
            lr=lr
        )

    def compute_loss(self, logprob_new, logprob_old, advantage):
        ratio = torch.exp(logprob_new - logprob_old)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
        return loss

    def train(self, epochs=1):
        self.policy_model.train()

        for epoch in range(epochs):
            total_loss = 0.0

            for batch in self.dataloader:
                prompts = batch["prompt"]

                # --- 1. Generate from policy model ---
                # You need tokenizer here
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                generated_outputs = self.policy_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False
                )
                generated_ids = generated_outputs[:, input_ids.shape[1]:]  # only new tokens

                # --- 2. Compute reward and value for generated sequence ---
                reward_score = self.reward_model(
                    input_ids=torch.cat([input_ids, generated_ids], dim=1),
                    attention_mask=torch.cat([attention_mask, torch.ones_like(generated_ids, device=self.device)],
                                             dim=1)
                )
                value_score = self.value_model(
                    input_ids=torch.cat([input_ids, generated_ids], dim=1),
                    attention_mask=torch.cat([attention_mask, torch.ones_like(generated_ids, device=self.device)],
                                             dim=1)
                )

                advantage = reward_score - value_score

                # --- 3. Compute logprobs ---
                outputs = self.policy_model(input_ids=torch.cat([input_ids, generated_ids], dim=1),
                                            attention_mask=torch.cat(
                                                [attention_mask, torch.ones_like(generated_ids, device=self.device)],
                                                dim=1),
                                            return_dict=True)
                logits = outputs.logits
                log_probs = F.log_softmax(logits, dim=-1)
                logprob_new = torch.gather(log_probs, -1, generated_ids.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

                # --- 4. Compute old logprobs with frozen reference ---
                with torch.no_grad():
                    outputs_ref = self.ref_model(input_ids=torch.cat([input_ids, generated_ids], dim=1),
                                                 attention_mask=torch.cat([attention_mask,
                                                                           torch.ones_like(generated_ids,
                                                                                           device=self.device)], dim=1),
                                                 return_dict=True)
                    logits_ref = outputs_ref.logits
                    log_probs_ref = F.log_softmax(logits_ref, dim=-1)
                    logprob_old = torch.gather(log_probs_ref, -1, generated_ids.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

                # --- 5. PPO loss and update ---
                loss = self.compute_loss(logprob_new, logprob_old, advantage)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch + 1} | Online PPO loss: {avg_loss:.4f}")
