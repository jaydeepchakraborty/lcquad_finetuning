from util.util_lib import *

from lcquad_finetuning.util.lcquad_util import LCQuadUtil

class GPTModelLoader:

    def __init__(self, config):
        self.config = config


    def load_gpt_model(self, pre_train_wt_ind=True):

        print(f'pre-trained model ind:- {self.config["model"]["chosen_model"]} START')
        model_obj = GPT2LMHeadModel.from_pretrained(self.config['model']['chosen_model'])
        tokenizer = LCQuadUtil.get_tokenizer(self.config)
        model_obj.resize_token_embeddings(len(tokenizer))

        device = self.config['model']['device']
        model_obj.to(device)
        print(f'pre-trained model ind:- {self.config["model"]["chosen_model"]} END')

        return model_obj