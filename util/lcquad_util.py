from lcquad_finetuning.util.util_lib import *



class LCQuadUtil:

    def __init__(self):
        # """
        # GPT-3,4 are not available
        # """
        # self.encodings = {
        #     "gpt2": tiktoken.get_encoding("gpt2"),
        #     # # "gpt3": tiktoken.get_encoding("p50k_base"),  # commonly associated with gpt-3 models
        #     # "gpt4": tiktoken.get_encoding("cl100k_base"),  # used for gpt-4 and later version
        #     # "gpt4o": tiktoken.get_encoding("o200k_base"),  # used for gpt-4 and later version
        # }
        pass

    @staticmethod
    def save_tokenizer(new_tokens, conf):
        tokenizer = GPT2Tokenizer.from_pretrained(conf['model']['tokenizer'])
        tokenizer.add_tokens(new_tokens)
        tokenizer.pad_token = tokenizer.eos_token # <|endoftext|>
        tokenizer.pad_token_id = tokenizer.eos_token_id  # usually 50256
        tokenizer.save_pretrained(conf["model"]["tokenizer_path"])

    @staticmethod
    def get_tokenizer(conf):
        tokenizer = GPT2Tokenizer.from_pretrained(conf["model"]["tokenizer_path"])
        return tokenizer

    @staticmethod
    def format_entry(entry, ind):
        instruction_text = (
            f"Question → SPARQL:"
        )

        question = f"\nQuestion: {entry['question']}" if entry["question"] else ""

        if ind == "train":
            sparql = f"\nSPARQL: {entry['sparql_modf']}" if entry["sparql_modf"] else ""
            ip_txt = instruction_text + question + sparql
        elif ind == "test":
            ip_txt = instruction_text + question
        else:
            raise NotImplementedError

        return ip_txt