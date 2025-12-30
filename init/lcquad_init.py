from lcquad_finetuning.init.lcquad_download_data import LCQuadDownloadData
from lcquad_finetuning.init.lcquad_download_model import LCQuadDownloadModel
from lcquad_finetuning.init.lcquad_tokens import LCQUADtokens


class LCQuadInit:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger


    def lcquad_init(self):
        # loading the base data
        lcquad_download_data = LCQuadDownloadData(self.config, self.logger)
        lcquad_download_data.download_data()

        # generating new tokens
        lcquad_tokens_obj = LCQUADtokens(self.config, self.logger)
        lcquad_tokens_obj.generate_tokens()

        # download model (decoder model/ causal ML model/ tokenizer)
        lcquad_download_model = LCQuadDownloadModel(self.config, self.logger)
        lcquad_download_model.populate_base_model_tokenizer()

        return