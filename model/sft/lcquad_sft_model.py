from lcquad_finetuning.util.lcquad_exception import LCQUADException
from lcquad_finetuning.util.util_lib import *


class LCQUADSFTModel:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def calc_loss_batch(self, input_batch, target_batch, model):
        outputs = model(input_ids=input_batch, labels=target_batch)
        loss = outputs.loss
        return loss

    def calc_loss_loader(self, dataloader, model):
        model.eval()
        with torch.no_grad():
            total_loss = 0
            if len(dataloader) == 0:
                return float("nan")

            num_batches = len(dataloader)

            for batch_data in dataloader:
                input_batch, target_batch = batch_data['ip_modf_token_ids'], batch_data['lbl_modf_token_ids']
                loss = self.calc_loss_batch(input_batch, target_batch, model)
                total_loss += loss.item()

        model.train()
        return total_loss / num_batches


    def train_lcquad_sft_model(self, model, train_loader, val_loader):

        device = self.config['model']['device']
        self.logger.info(f"device:- {device}")

        model = model.to(device)

        num_epochs = self.config['model']['sft_model']['model_config']['num_epochs']
        epoch_eval_freq = self.config['model']['sft_model']['model_config']['epoch_eval_freq']

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=5e-5
        )

        # Use linear warmup + cosine decay (or linear decay).
        # This dramatically improves stability and prevents catastrophic overfitting.
        num_training_steps = num_epochs * len(train_loader)
        num_warmup_steps = int(0.03 * num_training_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        effective_batch_size = self.config['model']['sft_model']['model_config']['batch_size']['effective_batch_size']  # what you WANT ~ 32
        real_batch_size = self.config['model']['sft_model']['model_config']['batch_size']['train_batch_size']  # what fits in RAM ~ 8
        accum_steps = effective_batch_size // real_batch_size

        for epoch in range(num_epochs):

            model.train() # set model to training mode
            optimizer.zero_grad()
            running_loss = 0.0

            for batch_id, batch_data in enumerate(train_loader):
                input_batch, target_batch = batch_data['ip_modf_token_ids'], batch_data['lbl_modf_token_ids']

                loss = self.calc_loss_batch(input_batch, target_batch, model)
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            if epoch % epoch_eval_freq == 0:
                train_loss = running_loss / len(train_loader)
                val_loss = self.calc_loss_loader(val_loader, model)
                self.logger.info(f"Epoch:- {epoch+1} Train loss:- {train_loss:3f} Val loss:- {val_loss:3f}")

        return model

