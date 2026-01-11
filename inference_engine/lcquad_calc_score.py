from lcquad_finetuning.util.util_lib import *

class LCQuadCalcScore:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def canonicalize(self, query):
        q = query.lower()
        q = re.sub(r'\s+', ' ', q).strip()
        return q

    def tokenize(self, s):
        return s.replace('\n', ' ').split()

    def token_prf(self, gold, pred):
        gold_tokens = self.tokenize(gold)
        pred_tokens = self.tokenize(pred)

        gold_set = set(gold_tokens)
        pred_set = set(pred_tokens)

        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        return precision, recall, f1

    def bleu_score(self, gold, pred):
        return sacrebleu.sentence_bleu(pred, [gold]).score

    def extract_triples(self, query):
        triples = []
        for line in query.split('.'):
            line = line.strip()
            if line.count(' ') >= 2:
                triples.append(tuple(line.split()[:3]))
        return set(triples)

    def triple_prf(self, gold, pred):
        gold_triples = self.extract_triples(gold)
        pred_triples = self.extract_triples(pred)

        if not gold_triples and not pred_triples:
            return 1.0, 1.0, 1.0

        tp = len(gold_triples & pred_triples)
        fp = len(pred_triples - gold_triples)
        fn = len(gold_triples - pred_triples)

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        return precision, recall, f1

    def save_result_scores(self, lcquad_result_df, score_msg):
        lcquad_result_df.to_csv(self.config['data']['inf_result_analysis_data'])
        with open(self.config['data']['inf_result_scores'], "w", encoding="utf-8") as f:
            f.write(score_msg)

    def lcquad_gen_scores(self, lcquad_result_df):

        gold_col = 'gold_sparql'
        pred_col = 'pred_sparql'

        # Canonical Exact Match
        lcquad_result_df['canonical_em'] = (
                lcquad_result_df[gold_col].apply(self.canonicalize) ==
                lcquad_result_df[pred_col].apply(self.canonicalize)
        ).astype(int)
        msg = f'Canonical Exact Match\n'
        msg = msg + f"Canonical EM:- {lcquad_result_df['canonical_em'].mean()} \n"

        # BLEU
        lcquad_result_df['bleu'] = lcquad_result_df.apply(lambda r: self.bleu_score(r[gold_col],
                                                                                    r[pred_col]), axis=1)
        msg = msg + f'BLEU Score\n'
        msg = msg + f"BLEU:- {lcquad_result_df['bleu'].mean()} \n"

        # Token - level Precision / Recall / F1
        lcquad_result_df[['token_precision', 'token_recall', 'token_f1']] = lcquad_result_df.apply(
            lambda r: pd.Series(self.token_prf(r[gold_col], r[pred_col])),
            axis=1
        )
        msg = msg + f'Token - level Precision / Recall / F1 \n'
        msg = msg + f"token_precision:- {lcquad_result_df['token_precision'].mean()} \n"
        msg = msg + f"token_recall:- {lcquad_result_df['token_recall'].mean()} \n"
        msg = msg + f"token_f1:- {lcquad_result_df['token_f1'].mean()} \n"

        # Triple-Pattern Precision / Recall
        lcquad_result_df[['triple_precision', 'triple_recall', 'triple_f1']] = lcquad_result_df.apply(lambda r: pd.Series(self.triple_prf(
            r[gold_col], r[pred_col])), axis=1)
        msg = msg + f'Triple-Pattern Precision / Recall \n'
        msg = msg + f"triple_precision:- {lcquad_result_df['triple_precision'].mean()} \n"
        msg = msg + f"triple_recall:- {lcquad_result_df['triple_recall'].mean()} \n"
        msg = msg + f"triple_f1:- {lcquad_result_df['triple_f1'].mean()} \n"

        self.save_result_scores(lcquad_result_df, msg)

