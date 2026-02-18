from lcquad_finetuning.util.util_lib import *
import lcquad_finetuning.util.lcquad_cnst as lcquad_cnst
from lcquad_finetuning.data.lcquad_datahelper import LCQUADDataHelper
from lcquad_finetuning.model.dapt.lcquad_dapt_model import LCQUADDAPTModel
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer
from lcquad_finetuning.util.lcquad_exception import LCQUADException
from lcquad_finetuning.util.lcquad_util import LCQuadUtil
from lcquad_finetuning.model.lcquad_modelhelper import LCQUADMODELHelper
from lcquad_finetuning.model.dapt.lcquad_dapt_testhelper import LCQUADDAPTMODELTESTHelper


class LCQUADDAPTMODELHelper:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def load_train_dapt_dataset(self):
        lcquad_train_dapt_obj = LCQUADDataHelper(self.config, self.logger)
        lcquad_train_dapt_ds = lcquad_train_dapt_obj.load_dapt_dataset()
        return lcquad_train_dapt_ds

    def load_tokenizer(self):
        lcquad_tokenizer_obj = LCQUADTokenizer(self.config, self.logger)
        tokenizer = lcquad_tokenizer_obj.load_tokenizer()
        return tokenizer

    def load_base_model(self):

        lcquad_modelhelper = LCQUADMODELHelper(self.config, self.logger)
        model_obj = lcquad_modelhelper.load_model("lcquad_base_model")

        return model_obj

    def save_lcquad_dapt_model(self, lcquad_dapt_model):

        lcquad_modelhelper = LCQUADMODELHelper(self.config, self.logger)
        lcquad_modelhelper.save_lcquad_dapt_model(lcquad_dapt_model)

        return


    def training_lcquad_dapt_model(self):

        # loading training data (only sparql)
        lcquad_train_dapt_ds = self.load_train_dapt_dataset()

        # loading the modified tokenizer
        tokenizer = self.load_tokenizer()
        tokenizer.padding_side = "right"  # for training

        # loading the base LLM model
        base_model = self.load_base_model()
        base_model.train()
        device = self.config['model']['device']
        base_model.to(device)

        # free any leftover CPU tensors and MPS cache after model loading
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        LCQuadUtil.log_mps_memory(self.logger, tag="after base_model.to(device)")

        lcquad_model = LCQUADDAPTModel(self.config, self.logger)
        # domain adaptive pretraining
        trainer = lcquad_model.train_lcquad_dapt_model(lcquad_train_dapt_ds, tokenizer, base_model)

        return trainer

    def load_dapt_model(self):
        lcquad_modelhelper = LCQUADMODELHelper(self.config, self.logger)
        model_obj = lcquad_modelhelper.load_model("lcquad_dapt_model")

        return model_obj

    def test_lcquad_dapt_model(self):

        # loading the modified tokenizer
        tokenizer = self.load_tokenizer()
        tokenizer.padding_side = "right"

        dapt_model = self.load_dapt_model()
        device = self.config['model']['device']
        dapt_model.to(device)
        dapt_model.eval()


        lcquad_model_test_obj = LCQUADDAPTMODELTESTHelper(self.config, self.logger)

        # validation on the single sample data dataset
        prefix = "SELECT ?answer WHERE { wd:Q4549135 "
        next_token = "wdt:P22 ?X"
        lcquad_model_test_obj.test_lcquad_dapt_model_with_prefix(prefix, next_token, tokenizer, dapt_model)

        # validation on the entire dataset
        lcquad_test_dapt_ds = self.load_train_dapt_dataset()
        lcquad_model_test_obj.test_lcquad_dapt_model_with_datatset(lcquad_test_dapt_ds, tokenizer, dapt_model)
