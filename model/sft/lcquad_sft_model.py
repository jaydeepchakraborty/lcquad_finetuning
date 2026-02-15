from lcquad_finetuning.util.lcquad_exception import LCQUADException
from lcquad_finetuning.util.util_lib import *


class LCQUADSFTModel:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def calc_loss_loader(self, dataloader, model):

        val_loss = 0.0

        model.eval()
        with ((torch.no_grad())):
            total_loss = 0
            if len(dataloader) == 0:
                return float("nan")

            num_batches = len(dataloader)

            for batch_data in dataloader:
                input_batch = batch_data['ip_modf_token_ids']
                attention_mask_batch = batch_data['attention_mask']
                target_batch = batch_data['lbl_modf_token_ids']

                outputs = model(input_ids=input_batch,
                                attention_mask=attention_mask_batch,
                                labels=target_batch)
                loss = outputs.loss
                total_loss += loss.item()

            val_loss = total_loss / num_batches

        model.train()
        return val_loss


    def train_lcquad_sft_model(self, model, train_loader, val_loader):

        num_epochs = self.config['model']['sft_model']['model_config']['num_epochs']
        epoch_eval_freq = self.config['model']['sft_model']['model_config']['epoch_eval_freq']

        effective_batch_size = self.config['model']['sft_model']['model_config']['batch_size']['effective_batch_size']  # what you WANT ~ 32
        real_batch_size = self.config['model']['sft_model']['model_config']['batch_size']['train_batch_size']  # what fits in RAM ~ 8
        accum_steps = effective_batch_size // real_batch_size

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=2e-5,
            betas=(0.9, 0.95),
            weight_decay=0.0
        )

        # Use linear warmup + cosine decay (or linear decay).
        # This dramatically improves stability and prevents catastrophic overfitting.
        # num_training_steps = number of optimizer.step() calls, not number of batches
        steps_per_epoch = -(-len(train_loader) // accum_steps)  # ceiling division
        num_training_steps = num_epochs * steps_per_epoch
        num_warmup_steps = int(0.03 * num_training_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        for epoch in range(num_epochs):

            model.train() # set model to training mode
            optimizer.zero_grad()
            running_loss = 0.0

            for batch_id, batch_data in enumerate(train_loader):
                input_batch = batch_data['ip_modf_token_ids']
                attention_mask_batch = batch_data['attention_mask']
                target_batch = batch_data['lbl_modf_token_ids']

                """
                attention_mask: 
                Tells model to ignore padding tokens (151668)
                Without it, model processes padding as real tokens
                target_batch:
                Computes loss automatically
                -100 tokens ignored in loss
                """
                outputs = model(input_ids=input_batch, # padded token IDs
                                attention_mask=attention_mask_batch, # 1 for real, 0 for PAD
                                labels=target_batch) # aligned with input_ids, -100 for prompt+PAD
                loss = outputs.loss
                running_loss += loss.item()
                loss = loss / accum_steps  # normalize loss
                loss.backward()

                """
                you cannot increase batch size due to memory limits, gradient accumulation is the correct and standard way 
                to reach an effective batch size large enough for stable Transformer training.
                """
                # Gradient Accumulation
                if (batch_id + 1) % accum_steps == 0:
                    # Clip Gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # Flush leftover accumulated gradients at epoch end
            if len(train_loader) % accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if epoch % epoch_eval_freq == 0:
                train_loss = running_loss / len(train_loader)
                val_loss = self.calc_loss_loader(val_loader, model)
                self.logger.info(f"Epoch:- {epoch+1} Train loss:- {train_loss:3f} Val loss:- {val_loss:3f}")

        return model

