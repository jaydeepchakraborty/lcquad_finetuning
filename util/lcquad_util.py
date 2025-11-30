from sympy.integrals.intpoly import strip

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
        print(f"loading tokenizer: {conf['model']['tokenizer']}")
        tokenizer = GPT2Tokenizer.from_pretrained(conf['model']['tokenizer'])
        print(f"pre-modified tokenizer {conf['model']['tokenizer']} with length {len(tokenizer)}")
        special = {"additional_special_tokens": ["<SPARQL>"], "pad_token": "<PAD>"}  # also add PAD if needed
        tokenizer.add_special_tokens(special)
        tokenizer.add_tokens(new_tokens)
        print(f"post-modified tokenizer {conf['model']['tokenizer']} with length {len(tokenizer)}")
        tokenizer.save_pretrained(conf["model"]["tokenizer_path"])
        print(f"saved tokenizer {conf['model']['tokenizer_path']}")

    @staticmethod
    def get_tokenizer(conf):
        print(f"loading tokenizer: {conf['model']['tokenizer_path']} - START")
        tokenizer = GPT2Tokenizer.from_pretrained(conf["model"]["tokenizer_path"])
        print(f"loading tokenizer: {conf['model']['tokenizer_path']} - FINISH")
        return tokenizer

    @staticmethod
    def normalize_text(text):
        if not text:
            return ""
        text = text.strip()
        text = re.sub(r'\r', '', text)  # remove windows CR
        text = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)  # remove indent
        return text

    @staticmethod
    def format_entry(entry, ind):
        instruction_text = ""

        question = f"Question: {LCQuadUtil.normalize_text(entry['question'])}\n" if entry["question"] else ""

        if ind == "train":
            sparql = f"<SPARQL>\n{LCQuadUtil.normalize_text(entry['sparql_modf'])}" if entry["sparql_modf"] else ""
            ip_txt = instruction_text + question + sparql
        elif ind == "test":
            ip_txt = instruction_text + question + "<SPARQL>\n"
        else:
            raise NotImplementedError

        return ip_txt