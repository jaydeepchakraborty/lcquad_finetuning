from lcquad_finetuning.data.lcquad_datahelper import LCQUADDataHelper
from lcquad_finetuning.model.rlhf.lcquad_rlhf_model import LCQUADRLHFModel
from lcquad_finetuning.util.lcquad_exception import LCQUADException
from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer

class LCQUADRLHFMODELHelper:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def load_policy_model(self):
        model_path = self.config['model']['rlhf_model_path']
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

    def load_ploicy_tokenizer(self):
        lcquad_tokenizer_obj = LCQUADTokenizer(self.config, self.logger)
        tokenizer = lcquad_tokenizer_obj.load_tokenizer()
        return tokenizer

    def load_policy_dataloder(self):
        lcquad_datahelper = LCQUADDataHelper(self.config, self.logger)
        dataloader = lcquad_datahelper.load_policy_dataloader()
        return dataloader

    def train_policy_model(self):
        # loading the policy model (i.e. SFT model)
        policy_model = self.load_policy_model()

        # loading the policy tokenizer
        policy_tokenizer = self.load_ploicy_tokenizer()

        # loading the policy train data
        policy_train_dataloader = self.load_policy_dataloder()

        # train the PPO model
        lcquad_rlhf_model_obj = LCQUADRLHFModel(self.config, self.logger)
        lcquad_rlhf_model = lcquad_rlhf_model_obj.train_lcquad_rlhf_model(policy_model, policy_tokenizer, policy_train_dataloader)

        # saving the LCQUAD model
        lcquad_rlhf_model_obj.save_lcquad_rlhf_model(lcquad_rlhf_model)




