from lcquad_finetuning.util.util_lib import *
import lcquad_finetuning.util.lcquad_cnst as lcquad_cnst
from lcquad_finetuning.data.lcquad_datahelper import LCQUADDataHelper
from lcquad_finetuning.model.clm.lcquad_clm_model import LCQUADCLMModel
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer
from lcquad_finetuning.util.lcquad_exception import LCQUADException
from lcquad_finetuning.util.lcquad_util import LCQuadUtil
from lcquad_finetuning.model.lcquad_modelhelper import LCQUADMODELHelper
from lcquad_finetuning.model.clm.lcquad_clm_testhelper import LCQUADCLMMODELTESTHelper


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

        if self.config['model']['chosen_model'] == lcquad_cnst.MODEL_GPT:
            lcquad_clm_model.save_model(model_path)
            self.logger.info(f"CLM model saved to {model_path}")
            model_path = model_path.replace('latest', LCQuadUtil.get_curr_tm())
            lcquad_clm_model.save_model(model_path)
            self.logger.info(f"CLM model saved to {model_path}")
        elif self.config['model']['chosen_model'] == lcquad_cnst.MODEL_QWEN:
            lcquad_clm_model.save_model(model_path)
            self.logger.info(f"CLM model saved to {model_path}")
            model_path = model_path.replace('latest', LCQuadUtil.get_curr_tm())
            lcquad_clm_model.save_model(model_path)
            self.logger.info(f"CLM model saved to {model_path}")
        elif self.config['model']['chosen_model'] == lcquad_cnst.MODEL_MISTRAL:
            lcquad_clm_model.save_model(model_path)
            self.logger.info(f"CLM model saved to {model_path}")
            model_path = model_path.replace('latest', LCQuadUtil.get_curr_tm())
            lcquad_clm_model.save_model(model_path)
            self.logger.info(f"CLM model saved to {model_path}")
        elif self.config['model']['chosen_model'] == lcquad_cnst.MODEL_GEMMA:
            lcquad_clm_model.save_model(model_path)
            self.logger.info(f"CLM model saved to {model_path}")
            model_path = model_path.replace('latest', LCQuadUtil.get_curr_tm())
            lcquad_clm_model.save_model(model_path)
            self.logger.info(f"CLM model saved to {model_path}")
        elif self.config['model']['chosen_model'] == lcquad_cnst.MODEL_LLAMA:
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
        tokenizer.padding_side = "right"  # for training

        # loading the base LLM model
        base_model = self.load_base_model()
        base_model.train()
        device = self.config['model']['device']
        base_model.to(device)

        lcquad_model = LCQUADCLMModel(self.config, self.logger)
        # domain adaptive pretraining
        trainer = lcquad_model.train_lcquad_clm_model(lcquad_train_clm_ds, tokenizer, base_model)

        return trainer

    def load_clm_model(self):
        lcquad_modelhelper = LCQUADMODELHelper(self.config, self.logger)
        model_obj = lcquad_modelhelper.load_model("lcquad_clm_model")

        return model_obj

    def test_lcquad_clm_model(self):

        # loading the modified tokenizer
        tokenizer = self.load_tokenizer()
        tokenizer.padding_side = "right"

        clm_model = self.load_clm_model()
        device = self.config['model']['device']
        clm_model.to(device)
        clm_model.eval()


        lcquad_model_test_obj = LCQUADCLMMODELTESTHelper(self.config, self.logger)

        # validation on the single sample data dataset
        prefix = "SELECT ?answer WHERE { wd:Q4549135 "
        next_token = "wdt:P22 ?X"
        lcquad_model_test_obj.test_lcquad_clm_model_with_prefix(prefix, next_token, tokenizer, clm_model)

        # validation on the entire dataset
        lcquad_test_clm_ds = self.load_train_clm_dataset()
        lcquad_model_test_obj.test_lcquad_clm_model_with_datatset(lcquad_test_clm_ds, tokenizer, clm_model)




