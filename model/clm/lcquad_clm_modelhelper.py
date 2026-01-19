from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.data.lcquad_datahelper import LCQUADDataHelper
from lcquad_finetuning.model.clm.lcquad_clm_model import LCQUADCLMModel
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer
from lcquad_finetuning.util.lcquad_exception import LCQUADException
from lcquad_finetuning.util.lcquad_util import LCQuadUtil
from lcquad_finetuning.model.lcquad_modelhelper import LCQUADMODELHelper


class LCQUADCLMMODELHelper:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def load_train_clm_dataset(self):
        lcquad_train_clm_obj = LCQUADDataHelper(self.config, self.logger)
        lcquad_train_clm_ds = lcquad_train_clm_obj.load_clm_dataset()
        return lcquad_train_clm_ds

    def load_tokenizer(self):
        lcquad_tokenizer_obj = LCQUADTokenizer(self.config, self.logger)
        tokenizer = lcquad_tokenizer_obj.load_tokenizer()
        return tokenizer

    def load_base_model(self):

        lcquad_modelhelper = LCQUADMODELHelper(self.config, self.logger)
        model_obj = lcquad_modelhelper.load_model("lcquad_base_model")

        return model_obj

    def save_lcquad_clm_model(self, lcquad_clm_model):

        model_path = self.config['model']['clm_model']['clm_model_path']

        if self.config['model']['chosen_model'] == "gpt2":
            lcquad_clm_model.save_pretrained(model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            lcquad_clm_model.save_model(model_path)
            self.logger.info(f"CLM model saved to {model_path}")
            model_path = model_path.replace('latest', LCQuadUtil.get_curr_tm())
            lcquad_clm_model.save_model(model_path)
            self.logger.info(f"CLM model saved to {model_path}")
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        return


    def training_lcquad_clm_model(self):

        # loading training data (only sparql)
        lcquad_train_clm_ds = self.load_train_clm_dataset()

        # loading the modified tokenizer
        tokenizer = self.load_tokenizer()

        # loading the base LLM model
        base_model = self.load_base_model()

        lcquad_model = LCQUADCLMModel(self.config, self.logger)
        # domain adaptive pretraining
        trainer = lcquad_model.train_lcquad_clm_model(lcquad_train_clm_ds, tokenizer, base_model)

        return trainer






