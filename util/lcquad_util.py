from lcquad_finetuning.util.util_lib import *

class LCQuadUtil:

    def __init__(self):
        """
        GPT-3,4 are not available
        """
        self.encodings = {
            "gpt2": tiktoken.get_encoding("gpt2"),
            # # "gpt3": tiktoken.get_encoding("p50k_base"),  # commonly associated with gpt-3 models
            # "gpt4": tiktoken.get_encoding("cl100k_base"),  # used for gpt-4 and later version
            # "gpt4o": tiktoken.get_encoding("o200k_base"),  # used for gpt-4 and later version
        }

        # # Load the base GPT-2 tokenizer
        # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # new_tokens = ["SPARQL", "ENTITY_01633", "ENTITY_03800"]
        # tokenizer.add_tokens(new_tokens)

    def get_tokenizer(self, tokenizer_id):
        # tiktoken
        return self.encodings[tokenizer_id]

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