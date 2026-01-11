from lcquad_finetuning.util.lcquad_util import LCQuadUtil
from lcquad_finetuning.util.util_lib import *

class LCQUADCLMModel:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def train_lcquad_clm_model(self, train_clm_dataset, tokenizer, base_model):

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        clm_model_path = self.config['model']['clm_model_path']
        clm_model_path = clm_model_path.replace('latest', 'tmp')
        training_args = TrainingArguments(output_dir=clm_model_path,
                                          overwrite_output_dir=True,
                                          per_device_train_batch_size=2,
                                          gradient_accumulation_steps=4,
                                          num_train_epochs=2,
                                          learning_rate=5e-5,
                                          warmup_steps=100,
                                          fp16=True,
                                          logging_steps=50,
                                          save_steps=500,
                                          save_total_limit=2)

        # training the model
        trainer = Trainer(
            model=base_model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_clm_dataset,  # Torch Dataset
        )

        self.logger.info(f"CLM trainer is saved at {clm_model_path}")

        return trainer

