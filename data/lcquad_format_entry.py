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
    def sft_format_entry_right_pad(entry, ind):
        instruction_text = ""

        question = f"Question: {LCQuadFormatEntry.normalize_text(entry['question'])}\n" if entry["question"] else ""

        if ind == "train":
            sparql = f"<SPARQL>\n{LCQuadFormatEntry.normalize_text(entry['sparql'])}" if entry["sparql"] else ""
            ip_txt = instruction_text + question + sparql
        elif ind == "test":
            ip_txt = instruction_text + question + "<SPARQL>\n"
        else:
            raise NotImplementedError

        return ip_txt

    @staticmethod
    def sft_format_entry_left_pad(entry, ind):
        instruction_text = ""

        question = f"Question: {LCQuadFormatEntry.normalize_text(entry['question'])}\n" if entry["question"] else ""

        if ind == "train":
            ip_txt = instruction_text + question + "<SPARQL>\n"
        elif ind == "test":
            ip_txt = instruction_text + question + "<SPARQL>\n"
        else:
            raise NotImplementedError

        return ip_txt

    @staticmethod
    def rm_format_entry(entry, ind):
        instruction_text = ""

        question = f"Question: {LCQuadFormatEntry.normalize_text(entry['question'])}\n" if entry["question"] else ""

        if ind == "train":
            sparql = f"<SPARQL>\n{LCQuadFormatEntry.normalize_text(entry['generated_sparql'])}" if entry["generated_sparql"] else ""
            ip_txt = instruction_text + question + sparql
        elif ind == "test":
            sparql = f"<SPARQL>\n{LCQuadFormatEntry.normalize_text(entry['generated_sparql'])}" if entry["generated_sparql"] else ""
            ip_txt = instruction_text + question + sparql
        else:
            raise NotImplementedError

        return ip_txt