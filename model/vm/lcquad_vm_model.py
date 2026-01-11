from lcquad_finetuning.util.util_lib import *

class LCQUADVMModel(nn.Module):
    def __init__(self, base_model, config, logger):
        super().__init__()
        self.base = base_model
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)

        nn.init.normal_(self.value_head.weight, mean=0.0, std=0.01)

        # Freeze base model (IMPORTANT)
        for p in self.base.parameters():
            p.requires_grad = False

        # Train only value head
        for p in self.value_head.parameters():
            p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        """
        input_ids:      [B, T]
        attention_mask: [B, T]
        """

        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # last_hidden_state: [B, T, H]
        hidden = outputs.last_hidden_state

        # mask: [B, T, 1]
        mask = attention_mask.unsqueeze(-1)

        # pooled: [B, H]
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        # values: [B]
        values = self.value_head(pooled).squeeze(-1)

        return values
