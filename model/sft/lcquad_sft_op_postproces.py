from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.util.lcquad_util import LCQuadUtil
from lcquad_finetuning.util.lcquad_exception import LCQUADException

class LCQUADSFTMODELPostProcessor:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def similarity(self, a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def repair_id(self,
            wrong_id,
            question,
            candidate_ids,
            id2label
    ):
        if wrong_id not in id2label:
            return wrong_id  # unknown, skip

        wrong_label = id2label[wrong_id]
        best_id = wrong_id
        best_score = 0.0

        for cid in candidate_ids:
            if cid not in id2label:
                continue
            score = self.similarity(wrong_label, id2label[cid])
            if score > best_score:
                best_score = score
                best_id = cid

        # threshold avoids over-aggressive replacement
        if best_score > 0.6:
            return best_id
        return wrong_id

    def repair_sparql(self,
            sparql,
            question,
            entity_candidates,
            relation_candidates,
            id2label
    ):
        repaired = sparql

        entities, relations = self.extract_ids(sparql)

        for eid in entities:
            new_eid = self.repair_id(
                eid,
                question,
                entity_candidates,
                id2label
            )
            repaired = repaired.replace(eid, new_eid)

        for rid in relations:
            new_rid = self.repair_id(
                rid,
                question,
                relation_candidates,
                id2label
            )
            repaired = repaired.replace(rid, new_rid)

        return repaired

    def post_process_sparql_fix(sparql, question):
        # Step 1: Fix syntax (example 2 & 3)
        sparql = fix_missing_predicates(sparql)
        sparql = fix_query_type(sparql)  # ask -> select

        # Step 2: Validate and fix entities (example 1)
        sparql = validate_entities(sparql, question)

        # Step 3: Execute and verify
        if not dry_run_valid(sparql):
            sparql = fallback_to_template(question)

        return sparql

    def post_process_sparql_fix_helper(self):
        pass