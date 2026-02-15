from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.util.lcquad_exception import LCQUADException
from lcquad_finetuning.util.lcquad_util import LCQuadUtil

class LCQUADDAPTMODELTESTHelper:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def true_token_rank(self, prefix, true_next, tokenizer, model):
        """
        Auto-regressive token-by-token rank check.
        - Tokenize true_next into individual tokens
        - For each token: predict, record rank, append to prefix
        Returns list of (token_str, rank) tuples
        """

        results = []
        current_ids = tokenizer.encode(prefix, add_special_tokens=False)

        true_ids = tokenizer.encode(true_next, add_special_tokens=False)
        for true_id in true_ids:
            enc = torch.tensor([current_ids], device=model.device)
            with torch.no_grad():
                logits = model(input_ids=enc).logits[0, -1]

            rank = (logits > logits[true_id]).sum().item() + 1
            token_str = tokenizer.decode([true_id])
            results.append((token_str, rank))

            current_ids.append(true_id)

        return results

    def test_lcquad_dapt_model_with_prefix(self, prefix, next_token, tokenizer, model):

        results = self.true_token_rank(
            prefix,
            next_token,
            tokenizer,
            model
        )
        print("token-by-token ranks:- ")
        for token_str, rank in results:
            print(f"  token: '{token_str}', rank: {rank}")

        avg_rank = np.mean([r for _, r in results])
        print(f"avg rank:- {avg_rank:.1f}")

        return

    def is_trivial_token(self, token, tokenizer):
        """
        Returns True if token is whitespace-only or punctuation-only.
        """
        if token.startswith("Ġ") or token.startswith("▁"):
            token = token[1:]

        # whitespace
        if token.strip() == "":
            return True

        # punctuation-only
        if token in {"{", "}", ".", "(", ")", ","}:
            return True

        return False

    def test_lcquad_dapt_model_with_datatset(self, test_ds, tokenizer, model):
        ranks = []
        device = self.config['model']['device']

        for sample in test_ds:
            input_ids = torch.tensor(sample["input_ids"], device=device).unsqueeze(0)
            attention_mask = torch.tensor(sample["attention_mask"], device=device).unsqueeze(0)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            logits = outputs.logits  # [1, seq_len, vocab]
            batch_seq_len = input_ids.size(1)

            i = 1
            while i < batch_seq_len:
                # it check handles padding. if pad comes the input ends
                if attention_mask[0, i] == 0:
                    break

                true_id = input_ids[0, i].item()
                true_token = tokenizer.convert_ids_to_tokens(true_id)

                # skip trivial tokens
                if self.is_trivial_token(true_token, tokenizer):
                    i += 1
                    continue

                token_logits = logits[0, i - 1]

                true_logit = token_logits[true_id]
                rank = (token_logits > true_logit).sum().item() + 1

                ranks.append(rank)
                i += 1

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
