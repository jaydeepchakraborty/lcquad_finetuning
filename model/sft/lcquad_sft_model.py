from lcquad_finetuning.util.lcquad_exception import LCQUADException
from lcquad_finetuning.util.util_lib import *


class LCQUADSFTModel:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def load_lcquad_clm_model(self):

        model_path = self.config['model']['clm_model_path']
        self.logger.info(f"loading model from {model_path}")

        if self.config['model']['chosen_model'] == "gpt2":
            model_obj = GPT2LMHeadModel.from_pretrained(model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            # for full supervised finetuning
            # model_obj = AutoModelForCausalLM.from_pretrained(model_path)
            # for QLoRA supervised finetuning
            """
            Apple MPS does not support 4-bit / 8-bit quantization
            So we are doing LoRA + fp16 for mac
            """
            # # Load model in 4-bit
            # bnb_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_compute_dtype=torch.bfloat16,
            #     bnb_4bit_use_double_quant=True
            # )
            # model_obj = AutoModelForCausalLM.from_pretrained(model_path,
            #                                                  quantization_config=bnb_config,)
            # Prepare for k-bit training
            # model_obj = prepare_model_for_kbit_training(model_obj)
            model_obj = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16,
                device_map=None
            )

            # Attach LoRA adapters
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen attention proj layers
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model_obj = get_peft_model(model_obj, lora_config)
            for name, param in model_obj.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
            model_obj.print_trainable_parameters()
            model_obj.enable_input_require_grads()
            # Enable gradient checkpointing
            model_obj.config.use_cache = False  # Required for checkpointing, On CPU or small GPUs
            model_obj.gradient_checkpointing_enable() # save memory, only if OOM ( Out of Memory )
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        return model_obj

    def save_lcquad_sft_model(self, lcquad_sft_model):
        model_path = self.config['model']['sft_model_path']

        if self.config['model']['chosen_model'] == "gpt2":
            lcquad_sft_model.save_pretrained(model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            lcquad_sft_model.save_pretrained(model_path)
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        self.logger.info(f"model saved to {model_path}")

        return

    def load_lcquad_sft_model(self):
        model_path = self.config['model']['sft_model_path']

        if self.config['model']['chosen_model'] == "gpt2":
            model_obj = GPT2LMHeadModel.from_pretrained(model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            model_obj = AutoModelForCausalLM.from_pretrained(model_path)
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        self.logger.info(f"model loaded from {model_path}")

        return model_obj


    def calc_loss_batch(self, input_batch, target_batch, model, device):
        outputs = model(input_ids=input_batch, labels=target_batch)
        loss = outputs.loss

        return loss

    def calc_loss_loader(self, dataloader, model, device):
        model.eval()
        with torch.no_grad():
            total_loss = 0
            if len(dataloader) == 0:
                return float("nan")

            num_batches = len(dataloader)

            for batch_data in dataloader:
                input_batch, target_batch = batch_data['ip_modf_token_ids'], batch_data['lbl_modf_token_ids']
                loss = self.calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item()

        model.train()
        return total_loss / num_batches


    def train_lcquad_sft_model(self, train_loader, val_loader):

        device = self.config['model']['device']
        self.logger.info(f"device:- {device}")

        model = self.load_lcquad_clm_model()
        model = model.to(device)

        num_epochs = self.config['model']['num_epochs']
        epoch_eval_freq = self.config['model']['epoch_eval_freq']

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

        effective_batch_size = self.config['model']['batch_size']['effective_batch_size']  # what you WANT ~ 32
        real_batch_size = self.config['model']['batch_size']['train_batch_size']  # what fits in RAM ~ 8
        accum_steps = effective_batch_size // real_batch_size

        for epoch in range(num_epochs):

            model.train() # set model to training mode
            optimizer.zero_grad()
            running_loss = 0.0

            for batch_id, batch_data in enumerate(train_loader):
                input_batch, target_batch = batch_data['ip_modf_token_ids'], batch_data['lbl_modf_token_ids']

                loss = self.calc_loss_batch(input_batch, target_batch, model, device)
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
                running_loss = 0.0
                val_loss = self.calc_loss_loader(val_loader, model, device)
                self.logger.info(f"Epoch:- {epoch+1} Train loss:- {train_loss:3f} Val loss:- {val_loss:3f}")

        return model

