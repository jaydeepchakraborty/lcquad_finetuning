from lcquad_finetuning.util.lcquad_exception import LCQUADException
from lcquad_finetuning.util.lcquad_util import LCQuadUtil
from lcquad_finetuning.util.util_lib import *
import lcquad_finetuning.util.lcquad_cnst as lcquad_cnst

class LCQUADCLMModel:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def train_lcquad_clm_model(self, train_clm_dataset, tokenizer, base_model):

        """
        DataCollator does (automatically)
        For each batch:
            1) pad to max length in batch
            2) convert to tensors

            create labels = input_ids (shifted internally)

        """
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        """
        per_device_train_batch_size = 2
        gradient_accumulation_steps = 16
        effective_batch_size = 2 × 16 × 1(mps device) = 32
        
        Dataset size = 24,074
        Micro-batch = 2
        Accumulation = 16 
        
        Number of micro-batches per epoch:
        Each micro-batch = one forward/backward pass.
        24,074 / 2 = 12,037 micro-batches per epoch
        12,037 / 16 ≈ 752 optimizer steps per epoch
        if you train for 10 epochs
        752 × 10 ≈ 7,520 optimizer steps total
        """

        clm_model_path = self.config['model']['clm_model']['clm_model_path']
        clm_model_path = clm_model_path.replace('latest', 'tmp')

        if self.config['model']['chosen_model'] == lcquad_cnst.MODEL_GPT:
            training_args = TrainingArguments(output_dir=clm_model_path,
                                              overwrite_output_dir=True,

                                              fp16=False,  # use fp32 for stability
                                              bf16=False,
                                              use_cpu=False,  # FORCE CPU (disables CUDA + MPS)

                                              per_device_train_batch_size=2,
                                              gradient_accumulation_steps=16,

                                              num_train_epochs=int(self.config['model']['clm_model']['model_config'][
                                                                       'num_train_epochs']),

                                              learning_rate=1e-5,  # full-param DAPT fp32
                                              weight_decay=0.01,  # regularization for full-param training

                                              max_grad_norm=1.0,  # gradient clipping

                                              warmup_steps=200,
                                              # For the first 200 optimizer steps, the learning rate is gradually increased from 0 → target learning rate.

                                              logging_steps=500,
                                              save_steps=2000,
                                              save_total_limit=2,

                                              dataloader_pin_memory=False,
                                              # pin_memory=True is useful for CUDA (NVIDIA GPUs)
                                              )

            # training the model
            trainer = Trainer(
                model=base_model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_clm_dataset,  # Torch Dataset
            )

            trainer.train()

            return trainer
        elif self.config['model']['chosen_model'] == lcquad_cnst.MODEL_QWEN:
            training_args = TrainingArguments(output_dir=clm_model_path,
                                              overwrite_output_dir=True,

                                              fp16=False, # use fp32 for stability
                                              bf16=False,
                                              use_cpu=False, # FORCE CPU (disables CUDA + MPS)

                                              per_device_train_batch_size=2,
                                              gradient_accumulation_steps=16,

                                              num_train_epochs=int(self.config['model']['clm_model']['model_config']['num_train_epochs']),

                                              learning_rate=1e-5, # full-param DAPT fp32
                                              weight_decay=0.01, # regularization for full-param training

                                              max_grad_norm=1.0, # gradient clipping

                                              warmup_steps=200, # For the first 200 optimizer steps, the learning rate is gradually increased from 0 → target learning rate.

                                              logging_steps=500,
                                              save_steps=2000,
                                              save_total_limit=2,

                                              dataloader_pin_memory=False, # pin_memory=True is useful for CUDA (NVIDIA GPUs)
                                              )

            # training the model
            trainer = Trainer(
                model=base_model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_clm_dataset,  # Torch Dataset
            )

            trainer.train()

            return trainer
        elif self.config['model']['chosen_model'] == lcquad_cnst.MODEL_MISTRAL:
            training_args = TrainingArguments(output_dir=clm_model_path,
                                              overwrite_output_dir=True,

                                              fp16=False,  # use fp32 for stability
                                              bf16=False,
                                              use_cpu=False,  # FORCE CPU (disables CUDA + MPS)

                                              per_device_train_batch_size=2,
                                              gradient_accumulation_steps=16,
                                              gradient_checkpointing=True,
                                              remove_unused_columns=False,  # IMPORTANT for PEFT

                                              num_train_epochs=int(self.config['model']['clm_model']['model_config'][
                                                                       'num_train_epochs']),

                                              learning_rate=2e-4,  # LoRA DAPT fp32
                                              weight_decay=0.0,  # not needed for LoRA

                                              max_grad_norm=1.0,  # gradient clipping

                                              warmup_steps=200,
                                              # For the first 200 optimizer steps, the learning rate is gradually increased from 0 → target learning rate.

                                              logging_steps=500,
                                              save_steps=2000,
                                              save_total_limit=2,

                                              dataloader_pin_memory=False,
                                              # pin_memory=True is useful for CUDA (NVIDIA GPUs)
                                              )

            # training the model
            trainer = Trainer(
                model=base_model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_clm_dataset,  # Torch Dataset
            )

            trainer.train()

            return trainer
        elif self.config['model']['chosen_model'] == lcquad_cnst.MODEL_GEMMA:
            training_args = TrainingArguments(output_dir=clm_model_path,
                                              overwrite_output_dir=True,

                                              fp16=False,  # use fp32 for stability
                                              bf16=False,
                                              use_cpu=False,  # FORCE CPU (disables CUDA + MPS)

                                              per_device_train_batch_size=2,
                                              gradient_accumulation_steps=16,
                                              gradient_checkpointing=True,
                                              remove_unused_columns=False,  # IMPORTANT for PEFT

                                              num_train_epochs=int(self.config['model']['clm_model']['model_config'][
                                                                       'num_train_epochs']),

                                              learning_rate=2e-4,  # LoRA DAPT fp32
                                              weight_decay=0.0,  # not needed for LoRA

                                              max_grad_norm=1.0,  # gradient clipping

                                              warmup_steps=200,
                                              # For the first 200 optimizer steps, the learning rate is gradually increased from 0 → target learning rate.

                                              logging_steps=500,
                                              save_steps=2000,
                                              save_total_limit=2,

                                              dataloader_pin_memory=False,
                                              # pin_memory=True is useful for CUDA (NVIDIA GPUs)
                                              )

            # training the model
            trainer = Trainer(
                model=base_model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_clm_dataset,  # Torch Dataset
            )

            trainer.train()

            return trainer
        elif self.config['model']['chosen_model'] == lcquad_cnst.MODEL_LLAMA:
            training_args = TrainingArguments(output_dir=clm_model_path,
                                              overwrite_output_dir=True,

                                              fp16=False, # use fp32 for stability
                                              bf16=False,
                                              use_cpu=False, # FORCE CPU (disables CUDA + MPS)

                                              per_device_train_batch_size=2,
                                              gradient_accumulation_steps=16,
                                              gradient_checkpointing=True,
                                              remove_unused_columns=False,  # IMPORTANT for PEFT

                                              num_train_epochs=int(self.config['model']['clm_model']['model_config']['num_train_epochs']),

                                              learning_rate=2e-4, # LoRA DAPT fp32
                                              weight_decay=0.0, # not needed for LoRA

                                              max_grad_norm=1.0, # gradient clipping

                                              warmup_steps=200, # For the first 200 optimizer steps, the learning rate is gradually increased from 0 → target learning rate.

                                              logging_steps=500,
                                              save_steps=2000,
                                              save_total_limit=2,

                                              dataloader_pin_memory=False, # pin_memory=True is useful for CUDA (NVIDIA GPUs)
                                              )

            # training the model
            trainer = Trainer(
                model=base_model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_clm_dataset,  # Torch Dataset
            )

            trainer.train()

            return trainer
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

