from lcquad_finetuning.util.util_lib import *

class LCQuadFormatEntry:

    @staticmethod
    def normalize_text(text):
        if not text:
            return ""
        if text is None:
            return ""
        # Handles NaN, None, np.nan, float('nan')
        if pd.isna(text):
            return ""
        text = text.strip()
        text = re.sub(r'\r', '', text)  # remove windows CR
        text = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)  # remove indent
        return text

    @staticmethod
    def prompt_format_entry(entry, ind):
        instruction_text = ""

        question = f"<Q_START>{LCQuadFormatEntry.normalize_text(entry['question'])}<Q_END>" if entry["question"] else ""

        if ind == "prompt_with_response":
            sparql = f"<SPARQL_START>{LCQuadFormatEntry.normalize_text(entry['sparql'])}<SPARQL_END>" if entry["sparql"] else ""
            ip_txt = instruction_text + question + sparql
        elif ind == "prompt_without_response":
            ip_txt = instruction_text + question + "<SPARQL_START>"
        else:
            raise NotImplementedError

        return ip_txt

    @staticmethod
    def prompt_rm_format_entry(entry):
        instruction_text = ""

        question = f"<Q_START>{LCQuadFormatEntry.normalize_text(entry['question'])}<Q_END>" if entry["question"] else ""
        sparql = f"<SPARQL_START>{LCQuadFormatEntry.normalize_text(entry['generated_sparql'])}<SPARQL_END>" if entry["generated_sparql"] else ""
        ip_txt = instruction_text + question + sparql

        return ip_txt
