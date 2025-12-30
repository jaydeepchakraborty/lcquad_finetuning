from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer
from lcquad_finetuning.util.lcquad_exception import LCQUADException

class LCQUADCLMModel:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def load_tokenizer(self):
        lcquad_tokenizer_obj = LCQUADTokenizer(self.config, self.logger)
        tokenizer = lcquad_tokenizer_obj.load_tokenizer()
        return tokenizer

    def load_base_model(self):

        model_ind = self.config["model"]["chosen_model"]
        model_path = self.config['model']['base_model_path']
        self.logger.info(f"loading model from {model_path}")
        if self.config['model']['chosen_model'] == "gpt2":
            model_obj = GPT2LMHeadModel.from_pretrained(model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            model_obj = AutoModelForCausalLM.from_pretrained(model_path)
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        return model_obj

    def save_lcquad_clm_model(self, lcquad_clm_model):
        model_ind = self.config["model"]["chosen_model"]
        model_path = self.config['model']['clm_model_path']

        if self.config['model']['chosen_model'] == "gpt2":
            lcquad_clm_model.save_pretrained(model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            lcquad_clm_model.save_model(model_path)
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        self.logger.info(f"model saved to {model_path}")
        return


    def train_lcquad_clm_model(self, train_clm_dataset):

        tokenizer = self.load_tokenizer()
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args = TrainingArguments(output_dir=self.config['model']['clm_model_path'],
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

        # loading the base model
        model = self.load_base_model()

        # training the model
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_clm_dataset,  # Torch Dataset
        )

        return trainer

