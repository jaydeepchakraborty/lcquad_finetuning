from util.util_lib import *

from lcquad_finetuning.util.lcquad_util import LCQuadUtil

class GPTModelLoader:

    def __init__(self, config):
        self.config = config


    def load_gpt_model(self, tokenizer, pre_train_wt_ind=True):

        print(f'GPT pre-trained model ind:- {self.config["model"]["chosen_model"]} START')
        model_obj = GPT2LMHeadModel.from_pretrained(self.config['model']['chosen_model'])

        print(f"GPT pre-trained model token resized to {len(tokenizer)}")
        """
        Entity tokens are not semantically close to English words
        Covariance from English words is irrelevant
        Mean-based initialization creates subtle clustering effects that confuse learning
        Random init → clean, stable, predictable gradient flow
        We are training from scratch on these entity embeddings anyway
        """
        model_obj.resize_token_embeddings(len(tokenizer), mean_resizing=False)

        device = self.config['model']['device']
        model_obj.to(device)
        print(f'GPT pre-trained model ind:- {self.config["model"]["chosen_model"]} END')

        return model_obj