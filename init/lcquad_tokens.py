from lcquad_finetuning.util.util_lib import *

class LCQUADtokens():
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def generate_tokens(self):

        train_df = pd.read_csv(self.config['data']['base_train_data'])

        test_df = pd.read_csv(self.config['data']['base_test_data'])

        token_df = pd.concat([train_df[['sparql']], test_df[['sparql']]], axis=0, ignore_index=True)

        # Regex that returns full strings like "wd:Q188920", "wdt:P31", "ps:P27"
        eid_pattern = re.compile(r'\b(?:wd|wdt|p|ps):[A-Za-z]\d+\b')

        lcquad_tokens = set()
        for sparql in token_df['sparql'].dropna():
            lcquad_tokens.update(eid_pattern.findall(sparql))

        lcquad_tokens_lst = list(lcquad_tokens)
        self.logger.info(f"total new tokens:- {len(lcquad_tokens_lst)}")
        # ["wd:Q188920", "wdt:P31", "ps:P27"]
        with open(self.config['data']['lcquad_token'], "w") as f:
            json.dump(lcquad_tokens_lst, f, indent=2)

        self.logger.info(f"LCQUAD tokens generated in {self.config['data']['lcquad_token']}")

