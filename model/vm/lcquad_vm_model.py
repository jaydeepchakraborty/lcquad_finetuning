from lcquad_finetuning.util.util_lib import *

class LCQUADVMModel(nn.Module):
    def __init__(self, base_model, config, logger):
        super().__init__()
        self.base = base_model

        device = config['model']['device']
        dtype = next(base_model.parameters()).dtype

        self.value_head = nn.Linear(base_model.config.hidden_size, 1).to(device=device, dtype=dtype)
        nn.init.normal_(self.value_head.weight, mean=0.0, std=0.01)

        # # IDEALLY: DON'T freeze base - should train during RLHF (PPO)
        # for p in self.base.parameters():
        #     p.requires_grad = True

        # Here, freeze base - should not train during RLHF (PPO)
        # train only Head parameters during RLHF (PPO)
        for p in self.base.parameters():
            p.requires_grad = False
        for p in self.value_head.parameters():
            p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        """
        input_ids:      [B, T]
        attention_mask: [B, T]
        """

        # Base model frozen
        with torch.no_grad():
            outputs = self.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            # hidden_states[-1]: [B, T, H]
            hidden = outputs.hidden_states[-1]

        # Applying trainable head to each position (no pooling!)
        values = self.value_head(hidden).squeeze(-1)  # [B, T]

        return values
