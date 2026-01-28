from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.util.lcquad_exception import LCQUADException
from lcquad_finetuning.util.lcquad_util import LCQuadUtil

class LCQUADCLMMODELTESTHelper:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def next_token_stats(self, text, tokenizer, model, top_k=10):
        enc = tokenizer(text, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits  # [1, seq_len, vocab]

        next_logits = logits[0, -1]  # last position
        probs = F.softmax(next_logits, dim=-1)

        topk = torch.topk(probs, top_k)

        tokens = tokenizer.convert_ids_to_tokens(topk.indices.tolist())
        return list(zip(tokens, topk.values.tolist()))

    def true_token_rank(self, prefix, true_next, tokenizer, model):
        enc = tokenizer(prefix, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}

        true_id = tokenizer.convert_tokens_to_ids(true_next)

        with torch.no_grad():
            logits = model(**enc).logits[0, -1]

        rank = (logits > logits[true_id]).sum().item() + 1
        return rank

    def test_lcquad_clm_model_with_prefix(self, prefix, next_token, tokenizer, model):
        nxt_tok_stats_info = self.next_token_stats(prefix,
                                                   tokenizer,
                                                   model)
        print("nxt_tok_stats_info:- ")
        print(nxt_tok_stats_info)

        rank = self.true_token_rank(
            prefix,
            next_token,
            tokenizer,
            model
        )
        print(f"rank:- {rank}")

        return

    def is_trivial_token(self, token, tokenizer):
        """
        Returns True if token is whitespace-only or punctuation-only.
        """
        if token.startswith("Ġ"):
            token = token[1:]

        # whitespace
        if token.strip() == "":
            return True

        # punctuation-only
        if token in {"{", "}", ".", "(", ")", ","}:
            return True

        return False

    def test_lcquad_clm_model_with_datatset(self, test_ds, tokenizer, model):
        ranks = []
        device = self.config['model']['device']

        for sample in test_ds:
            input_ids = torch.tensor(sample["input_ids"], device=device).unsqueeze(0)
            attention_mask = torch.tensor(sample["attention_mask"], device=device).unsqueeze(0)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits  # [1, seq_len, vocab]
            seq_len = input_ids.size(1)

            for i in range(3, seq_len):
                if attention_mask[0, i] == 0:
                    break

                true_id = input_ids[0, i].item()
                true_token = tokenizer.convert_ids_to_tokens(true_id)

                # skip trivial tokens
                if self.is_trivial_token(true_token, tokenizer):
                    continue

                token_logits = logits[0, i - 1]

                true_logit = token_logits[true_id]
                rank = (token_logits > true_logit).sum().item() + 1

                ranks.append(rank)

        ranks = np.array(ranks)
        """
        Hit@1 = 82.2%
        In ~82.2% of non-trivial positions, the model’s top-1 prediction is exactly the true next token.
        Hit@5 = 88.1%
        In ~88.1% of cases, the correct token is within the top 5 predictions.
        Mean Rank = 523
        the average position of the true next token in the model’s predicted list.
        """
        metrics = {
            "Hit@1": np.mean(ranks <= 1),
            "Hit@5": np.mean(ranks <= 5),
            "Mean Rank": np.mean(ranks),
        }

        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
