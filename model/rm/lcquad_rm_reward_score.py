from lcquad_finetuning.util.util_lib import *

class LCQUADRMRewardScoreGenerator:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def nli_violation(self, row):
        org, gen = row['original_sparql'], row['generated_sparql']
        similarity = SequenceMatcher(None, org, gen).ratio()
        violation_score = round((1.0 - similarity), 2)
        return violation_score  # higher = worse

    def extract_terms(self, str_val):
        IMPORTANT_TOKENS = {"wd:", "wdt:", "p:", "ps:", "pq:",
                            "SELECT", "WHERE", "FILTER", "OPTIONAL",
                            "GROUP", "BY", "ASK", "COUNT", "LIMIT", "OFFSET"
                            "{", "}", ".", "?"}

        tokens = re.findall(r"\b\w+:\w+\b", str_val)
        keywords = set(tokens) | {t for t in IMPORTANT_TOKENS if t in str_val}
        return keywords

    def term_violation(self, row):
        org, gen = row['original_sparql'], row['generated_sparql']
        org_terms = self.extract_terms(org)
        gen_terms = self.extract_terms(gen)

        if not org_terms:
            return 0.0

        missing = org_terms - gen_terms
        violation_score = round((len(missing) / len(org_terms)), 2)

        return violation_score

    def extract_numbers(self, s_val):
        return set(re.findall(r"\d+", s_val))

    def numeric_mismatch(self, row):
        org, gen = row['original_sparql'], row['generated_sparql']
        org_nums = self.extract_numbers(org)
        gen_nums = self.extract_numbers(gen)

        if not org_nums:
            return 0.0

        mismatched = org_nums.symmetric_difference(gen_nums)

        mismatch_score = round((len(mismatched) / len(org_nums)), 2)

        return mismatch_score

    def attention_gap(self, row):
        org, gen = row['original_sparql'], row['generated_sparql']
        org_len = len(org.split())
        gen_len = len(gen.split())

        if org_len == 0:
            return 0.0

        ratio = gen_len / org_len
        if ratio < 0.5 or ratio > 1.5:
            return  round(min(1.0, abs(1 - ratio)), 2)

        return 0.0

    # def ood_score(self, row):
    #     org, gen = row['original_sparql'], row['generated_sparql']
    #     required = ["SELECT", "WHERE", "{", "}"]
    #     missing = [k for k in required if k not in gen.upper()]
    #     return len(missing) / len(required)

    def compute_violation_scores(self, df):

        df["nli_violation"] = df.apply(
            lambda r: self.nli_violation(r), axis=1
        )
        df["term_violation"] = df.apply(
            lambda r: self.term_violation(r), axis=1
        )
        df["numeric_mismatch"] = df.apply(
            lambda r: self.numeric_mismatch(r), axis=1
        )
        df["attention_gap"] = df.apply(
            lambda r: self.attention_gap(r), axis=1
        )
        # df["ood_score"] = df.apply(
        #     lambda r: self.ood_score(r), axis=1
        # )
        df["ood_score"] = 0 # means bad ~ out of distribution

        return df

    def compute_reward_score(self, row, weights=None):

        nli_violation = row['nli_violation']
        term_violation = row['term_violation']
        numeric_mismatch = row['numeric_mismatch']
        attention_gap = row['attention_gap']
        ood_score = row['ood_score']

        # default weights
        if weights is None:
            weights = {
                "nli": 0.35,
                "term": 0.25,
                "numeric": 0.20,
                "attention": 0.10,
                "ood": 0.10
            }

        penalty = (
                weights["nli"] * float(nli_violation) +
                weights["term"] * float(term_violation) +
                weights["numeric"] * float(numeric_mismatch) +
                weights["attention"] * float(attention_gap) +
                weights["ood"] * float(ood_score)
        )

        reward_score = 1.0 - penalty
        reward_score = max(0.0, min(1.0, reward_score))
        reward_score = round(reward_score, 2)

        return reward_score

    def compute_reward_score_util(self, df):
        df["reward_score"] = df.apply(
            lambda r: self.compute_reward_score(r), axis=1
        )
        return df

    def save_reward_score(self, df, data_fl_path):
        df.to_csv(data_fl_path, index=False)
        self.logger.info(f"Save reward score to {data_fl_path}")
        return

    def generate_reward_score(self, df, data_fl_path):

        # compute violation scores
        df = self.compute_violation_scores(df)

        # calculate the reward score
        df = self.compute_reward_score_util(df)

        # save the reward score
        self.save_reward_score(df, data_fl_path)

        return