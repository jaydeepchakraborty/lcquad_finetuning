import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

class HybridPPOTrainer:
    """
    Hybrid PPO trainer:
    - Uses precomputed reward (offline)
    - Computes value model and log-probabilities online
    - Updates LoRA weights of policy model
    """

    def __init__(
        self,
        policy_model,
        ref_model,
        value_model,
        optimizer,
        tokenizer,
        clip_eps=0.2,
        max_grad_norm=1.0,
        device="cuda"
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.value_model = value_model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.clip_eps = clip_eps
        self.max_grad_norm = max_grad_norm
        self.device = device

        # freeze reference and value models
        self.ref_model.eval()
        self.value_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
        for p in self.value_model.parameters():
            p.requires_grad = False

    def compute_logprobs(self, logits, input_ids, pad_token_id):
        """
        Compute log-probabilities for each token in input_ids
        """
        # logits: [B, T, V], input_ids: [B, T]
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        # gather log-probs of actual tokens
        log_probs_token = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
        # mask padding tokens
        mask = (input_ids != pad_token_id).float()
        log_probs_masked = log_probs_token * mask
        # sum per sequence
        seq_logprob = log_probs_masked.sum(dim=-1)
        return seq_logprob  # [B]

    def train_batch(self, batch):
        """
        batch: dict
            - prompt: list[str]
            - generated_sparql: list[str]
            - reward_score: torch.tensor [B]
        """
        prompts = batch["prompt"]
        responses = batch["generated_sparql"]
        rewards = batch["reward_score"].to(self.device)  # precomputed

        # Tokenize prompts + responses
        enc = self.tokenizer(prompts, responses, padding=True, return_tensors="pt").to(self.device)
        input_ids = enc.input_ids
        attention_mask = enc.attention_mask
        pad_token_id = self.tokenizer.pad_token_id

        # --- logprob_old from frozen reference model ---
        with torch.no_grad():
            ref_logits = self.ref_model(input_ids=input_ids, attention_mask=attention_mask).logits
            logprob_old = self.compute_logprobs(ref_logits, input_ids, pad_token_id)

        # --- logprob_new from current policy ---
        policy_logits = self.policy_model(input_ids=input_ids, attention_mask=attention_mask).logits
        logprob_new = self.compute_logprobs(policy_logits, input_ids, pad_token_id)

        # --- value from value model (online) ---
        with torch.no_grad():
            value = self.value_model(input_ids=input_ids, attention_mask=attention_mask)

        # --- advantage ---
        advantage = rewards - value  # [B]

        # --- PPO surrogate loss ---
        ratio = torch.exp(logprob_new - logprob_old)
        surrogate1 = ratio * advantage
        surrogate2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
        loss = -torch.mean(torch.min(surrogate1, surrogate2))

        # --- Backprop and optimization ---
        loss.backward()
        clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item(), advantage.mean().item()
