from lcquad_finetuning.util.util_lib import *

class LCQUADRMModel(nn.Module):

    def __init__(self, base_model, config, logger):
        super().__init__()
        self.model = base_model

        # freezing the base mode parameters
        for p in self.model.parameters():
            p.requires_grad = False

        device = config['model']['device']
        dtype = next(base_model.parameters()).dtype

        self.head = nn.Linear(self.model.config.hidden_size, 1).to(device=device, dtype=dtype)
        for p in self.head.parameters():
            p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        """
        input_ids:      [B, T]
        attention_mask: [B, T]
        """
        with torch.no_grad():  # base model frozen
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            # hidden_states[-1]: [B, T, H]
            hidden = outputs.hidden_states[-1]
        # mask: [B, T, 1]
        mask = attention_mask.unsqueeze(-1)
        mask = mask.to(hidden.dtype)
        denom = mask.sum(dim=1).clamp(min=1)  # prevent divide-by-zero

        # pooled: [B, H]
        pooled = (hidden * mask).sum(dim=1) / denom

        # reward: [B]
        reward = self.head(pooled).squeeze(-1)

        return reward
