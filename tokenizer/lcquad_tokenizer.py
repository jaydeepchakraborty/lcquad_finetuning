from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.util.lcquad_exception import LCQUADException


class LCQUADTokenizer:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def load_tokenizer(self):

        if self.config['model']['chosen_model'] == "gpt2":

            tokenizer_path = self.config["model"]["tokenizer_path"].replace("{model_ind}",
                                                                            f"{self.config['model']['chosen_model']}")
            self.logger.info(f"loading tokenizer: {tokenizer_path} - START")
            tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
            self.logger.info(f"loading tokenizer: {tokenizer_path} - FINISH")

        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":

            tokenizer_path = self.config["model"]["tokenizer_path"].replace("{model_ind}",
                                                                            f"{self.config['model']['chosen_model']}")
            self.logger.info(f"loading tokenizer: {tokenizer_path} - START")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.logger.info(f"loading tokenizer: {tokenizer_path} - FINISH")

        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        return tokenizer

    def lcquad_tok_decoder(self, toks, tokenizer):

        decoded_parts = []
        ignore_index = -100
        for tok in toks:
            if tok == ignore_index:
                decoded_parts.append("<IGNORE>")
            else:
                # Decode a single tokenizer safely
                text = tokenizer.decode([tok], skip_special_tokens=False)
                decoded_parts.append(text)

        return decoded_parts

    def lcquad_txt_encoder(self, txt, tokenizer):

        toks = tokenizer(
            txt,
            padding=False,
            return_tensors=None,
            add_special_tokens=False
        )

        return toks

