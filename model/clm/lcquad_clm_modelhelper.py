from lcquad_finetuning.data.lcquad_datahelper import LCQUADDataHelper
from lcquad_finetuning.model.clm.lcquad_clm_model import LCQUADCLMModel

class LCQUADCLMMODELHelper:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def load_train_clm_dataset(self):
        lcquad_train_clm_obj = LCQUADDataHelper(self.config, self.logger)
        lcquad_train_clm_ds = lcquad_train_clm_obj.load_clm_dataset()
        return lcquad_train_clm_ds

    def training_lcquad_clm_model(self):

        # loading training data (only sparql)
        lcquad_train_clm_ds = self.load_train_clm_dataset()

        lcquad_model = LCQUADCLMModel(self.config, self.logger)
        # domain adaptive pretraining
        trainer = lcquad_model.train_lcquad_clm_model(lcquad_train_clm_ds)
        # saving the model
        lcquad_model.save_lcquad_clm_model(trainer)

        return






