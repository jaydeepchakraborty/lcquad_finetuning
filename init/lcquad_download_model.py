from lcquad_finetuning.util.util_lib import *
import lcquad_finetuning.util.lcquad_cnst as lcquad_cnst
import lcquad_finetuning.tokens.lcquad_tokens as lcquad_tokens
from lcquad_finetuning.util.lcquad_exception import LCQUADException


class LCQuadDownloadModel:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def get_new_token_lst(self):
        new_tokens = {}
        self.logger.info(f"new tokens are loaded from {self.config['data']['lcquad_token']}")
        with open(self.config['data']['lcquad_token'], "r") as f:
            new_tokens = json.load(f)
        return list(new_tokens)

    def get_special_tokens(self):
        special_tokens = [
                    "<Q_START>", "<Q_END>",
                    "<SPARQL_START>", "<SPARQL_END>"
        ]
        return special_tokens

    def populate_base_tokenizer(self):

        self.logger.info(f"downloading base tokenizer: {self.config['model']['tokenizer']}")
        if self.config['model']['chosen_model'] == lcquad_cnst.MODEL_GPT:
            tokenizer = AutoTokenizer.from_pretrained(self.config['model']['tokenizer'],
                                                      token=lcquad_tokens.HUGGINGFACE_TOKEN)
        elif self.config['model']['chosen_model'] == lcquad_cnst.MODEL_QWEN:
            """
            By default, Qwen/Qwen2.5-1.5B tokenizer use "<|endoftext|>" for padding. 
            no need to update the "eos_token", "eos_token_id"
            but as we need left padding (during inference), we need separate "<PAD>" token.
            """
            tokenizer = AutoTokenizer.from_pretrained(self.config['model']['tokenizer'],
                                                            token=lcquad_tokens.HUGGINGFACE_TOKEN)
        elif self.config['model']['chosen_model'] == lcquad_cnst.MODEL_MISTRAL:
            """
            Mistral has no native PAD token.
            Mistral ~ train, inference both are right padding
            """
            tokenizer = AutoTokenizer.from_pretrained(self.config['model']['tokenizer'], use_fast=True,
                                                            token=lcquad_tokens.HUGGINGFACE_TOKEN)
        elif self.config['model']['chosen_model'] == lcquad_cnst.MODEL_GEMMA:
            """
            Gemma expects BOS at start.
            Explicitly prepend BOS once during training and inference.
            during tokenization:
                tokenizer(..., add_special_tokens=True)
            """
            tokenizer = AutoTokenizer.from_pretrained(self.config['model']['tokenizer'],
                                                      token=lcquad_tokens.HUGGINGFACE_TOKEN)
        elif self.config['model']['chosen_model'] == lcquad_cnst.MODEL_LLAMA:
            tokenizer = AutoTokenizer.from_pretrained(self.config['model']['tokenizer'],
                                                      token=lcquad_tokens.HUGGINGFACE_TOKEN)
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        self.logger.info(f"pre-modified tokenizer {self.config['model']['tokenizer']} with length {len(tokenizer)}")
        new_tokens = self.get_new_token_lst()
        tokenizer.add_tokens(new_tokens)

        special = {"additional_special_tokens": self.get_special_tokens(),
                   "pad_token": "<PAD>"}
        tokenizer.add_special_tokens(special)

        tokenizer_path = self.config["model"]["tokenizer_path"]
        self.logger.info(f"post-modified tokenizer {self.config['model']['tokenizer']} with length {len(tokenizer)}")
        tokenizer.save_pretrained(tokenizer_path)
        self.logger.info(f"saved tokenizer to {tokenizer_path}")


    def populate_base_model(self):

        if self.config['model']['chosen_model'] == lcquad_cnst.MODEL_GPT:

            self.logger.info(f'pre-trained Basemodel ind:- {self.config["model"]["chosen_model"]} START')
            model_obj = AutoModelForCausalLM.from_pretrained(self.config["model"]["chosen_model"],
                                                             token=lcquad_tokens.HUGGINGFACE_TOKEN)

            tokenizer_path = self.config["model"]["tokenizer_path"]
            self.logger.info(f"loading tokenizer: {tokenizer_path} - START")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.logger.info(f"loading tokenizer: {tokenizer_path} - FINISH")

        elif self.config['model']['chosen_model'] == lcquad_cnst.MODEL_QWEN:
            self.logger.info(f'pre-trained Basemodel ind:- {self.config["model"]["chosen_model"]} START')
            model_obj = AutoModelForCausalLM.from_pretrained(self.config["model"]["chosen_model"],
                                                            token=lcquad_tokens.HUGGINGFACE_TOKEN)

            tokenizer_path = self.config["model"]["tokenizer_path"]
            self.logger.info(f"loading tokenizer: {tokenizer_path} - START")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.logger.info(f"loading tokenizer: {tokenizer_path} - FINISH")
        elif self.config['model']['chosen_model'] == lcquad_cnst.MODEL_MISTRAL:
            self.logger.info(f'pre-trained Basemodel ind:- {self.config["model"]["chosen_model"]} START')
            model_obj = AutoModelForCausalLM.from_pretrained(self.config["model"]["chosen_model"],
                                                            token=lcquad_tokens.HUGGINGFACE_TOKEN)

            tokenizer_path = self.config["model"]["tokenizer_path"]
            self.logger.info(f"loading tokenizer: {tokenizer_path} - START")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True,)
            self.logger.info(f"loading tokenizer: {tokenizer_path} - FINISH")
        elif self.config['model']['chosen_model'] == lcquad_cnst.MODEL_GEMMA:
            self.logger.info(f'pre-trained Basemodel ind:- {self.config["model"]["chosen_model"]} START')
            model_obj = AutoModelForCausalLM.from_pretrained(self.config["model"]["chosen_model"],
                                                                token=lcquad_tokens.HUGGINGFACE_TOKEN)

            tokenizer_path = self.config["model"]["tokenizer_path"]
            self.logger.info(f"loading tokenizer: {tokenizer_path} - START")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.logger.info(f"loading tokenizer: {tokenizer_path} - FINISH")
        elif self.config['model']['chosen_model'] == lcquad_cnst.MODEL_LLAMA:
            self.logger.info(f'pre-trained Basemodel ind:- {self.config["model"]["chosen_model"]} START')
            model_obj = AutoModelForCausalLM.from_pretrained(self.config["model"]["chosen_model"],
                                                            token=lcquad_tokens.HUGGINGFACE_TOKEN)

            tokenizer_path = self.config["model"]["tokenizer_path"]
            self.logger.info(f"loading tokenizer: {tokenizer_path} - START")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.logger.info(f"loading tokenizer: {tokenizer_path} - FINISH")
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        self.logger.info(f"pre-trained Basemodel token resized to {len(tokenizer)}")
        model_obj.resize_token_embeddings(len(tokenizer), mean_resizing=True)

        model_path = self.config['model']['base_model_path']
        model_obj.save_pretrained(model_path)
        self.logger.info(f"pre-trained Basemodel saved to {model_path}")

    def populate_base_model_tokenizer(self):

        # populating base tokenizer
        self.populate_base_tokenizer()

        # populating base model
        self.populate_base_model()