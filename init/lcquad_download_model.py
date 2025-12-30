from lcquad_finetuning.util.util_lib import *
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
            "<SPARQL>", "SELECT", "WHERE", "FILTER", "OPTIONAL",
            "GROUP", "BY", "ASK", "COUNT", "LIMIT", "OFFSET"
            "{", "}", ".", "?"
        ]
        return special_tokens

    def populate_base_tokenizer(self):

        self.logger.info(f"downloading base tokenizer: {self.config['model']['tokenizer']}")
        if self.config['model']['chosen_model'] == "gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained(self.config['model']['tokenizer'])
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            tokenizer = AutoTokenizer.from_pretrained(self.config['model']['tokenizer'])
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        self.logger.info(f"pre-modified tokenizer {self.config['model']['tokenizer']} with length {len(tokenizer)}")
        new_tokens = self.get_new_token_lst()
        special = {"additional_special_tokens": self.get_special_tokens(),
                   "pad_token": "<PAD>"}
        tokenizer.add_special_tokens(special)
        tokenizer.add_tokens(new_tokens)
        tokenizer_path = self.config["model"]["tokenizer_path"].replace("{model_ind}",
                                                                        f"{self.config['model']['chosen_model']}")
        self.logger.info(f"post-modified tokenizer {self.config['model']['tokenizer']} with length {len(tokenizer)}")
        tokenizer.save_pretrained(tokenizer_path)
        self.logger.info(f"saved tokenizer to {tokenizer_path}")


    def populate_base_model(self):

        if self.config['model']['chosen_model'] == "gpt2":

            self.logger.info(f'pre-trained Basemodel ind:- {self.config["model"]["chosen_model"]} START')
            model_obj = GPT2LMHeadModel.from_pretrained(self.config['model']['chosen_model'])

            tokenizer_path = self.config["model"]["tokenizer_path"].replace("{model_ind}",
                                                                            f"{self.config['model']['chosen_model']}")
            self.logger.info(f"loading tokenizer: {tokenizer_path} - START")
            tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
            self.logger.info(f"loading tokenizer: {tokenizer_path} - FINISH")

        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            self.logger.info(f'pre-trained Basemodel ind:- {self.config["model"]["chosen_model"]} START')
            model_obj = AutoModelForCausalLM.from_pretrained(self.config["model"]["chosen_model"])

            tokenizer_path = self.config["model"]["tokenizer_path"]
            self.logger.info(f"loading tokenizer: {tokenizer_path} - START")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.logger.info(f"loading tokenizer: {tokenizer_path} - FINISH")

        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        self.logger.info(f"pre-trained Basemodel token resized to {len(tokenizer)}")
        model_obj.resize_token_embeddings(len(tokenizer), mean_resizing=False)

        model_path = self.config['model']['base_model_path']
        model_obj.save_pretrained(model_path)
        self.logger.info(f"pre-trained Basemodel saved to {model_path}")

    def populate_base_model_tokenizer(self):

        # populating base tokenizer
        self.populate_base_tokenizer()

        # populating base model
        self.populate_base_model()