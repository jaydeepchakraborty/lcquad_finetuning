from lcquad_finetuning.util.util_lib import *
class LCQuadConfig:
    def __init__(self,):

        current_datetime = datetime.now()
        timestamp_str = current_datetime.strftime("YR-%Y_MM-%m_DD-%d_HR-%H_M-%M_SEC-%S")  # e.g., 2025_10_21_20_05_00
        print(f"running for {timestamp_str}")

        BASE_PATH = "/Volumes/Jay_4TB/"
        DATA_PATH = BASE_PATH + "data/LC_Quad/"
        MODEL_PATH = BASE_PATH + "model_utils/models/LC_Quad/"
        # GPT_MODEL_PATH = BASE_PATH + "model_utils/models/llm/"

        self.lcquad_config = {
            "data":{
                    "train_data": DATA_PATH + "train.csv",
                    "test_data": DATA_PATH + "test.csv",
                    "sparql_wikidata_eids_labels_mapping": DATA_PATH + "sparql_wikidata_eid_label_map.json",
                    "sparql_wikidata_labels_eids_mapping": DATA_PATH + "sparql_wikidata_label_eid_map.json",
                    "new_token": DATA_PATH + "new_token.json",
                    "modf_train_data": DATA_PATH + "modf_train_data.csv",
                    "modf_valid_data": DATA_PATH + "modf_valid_data.csv",
                    "modf_test_data": DATA_PATH + "modf_test_data.csv",
                    "train_dataset": DATA_PATH + "train_dataset.pt",
                    "val_dataset": DATA_PATH + "valid_dataset.pt",
                    "test_dataset": DATA_PATH + "test_dataset.pt",
                },
            "model": {
                "chosen_model": "gpt2",
                "tokenizer": "gpt2",
                "tokenizer_path": MODEL_PATH + "lcquad_tokenizer",
                "gpt_config": {
                    "name": "gpt2-transformer",
                    "basic_config": {
                        "allowed_max_length": 1024, # context length
                        "pad_token_id": 50256, # <|endoftext|>
                        "ignore_index": -100
                    },
                },
                "device": None,
                "num_workers": 0,
                "batch_size": {
                    "train_batch_size": 8,
                    "test_batch_size": 4,
                    "val_batch_size": 4,
                },
                "model_path": MODEL_PATH + "lcquad_model_{model_ind}_" + str(timestamp_str),
                "inf_model_path": MODEL_PATH + "lcquad_model_{model_ind}",
                "num_epochs": 10,
                "eval_freq": 500,
            }
        }
        return

    def get_config(self,):

        # self.lcquad_config['model']['device'] = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.lcquad_config['model']['device'] = "cpu"

        return self.lcquad_config