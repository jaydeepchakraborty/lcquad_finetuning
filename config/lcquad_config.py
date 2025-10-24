from lcquad_finetuning.util.util_lib import *
class LCQuadConfig:
    def __init__(self,):

        current_datetime = datetime.now()
        timestamp_str = current_datetime.strftime("YR-%Y_MM-%m_DD-%d_HR-%H_M-%M_SEC-%S")  # e.g., 2025_10_21_20_05_00
        print(f"running for {timestamp_str}")

        BASE_PATH = "/Volumes/Jay_4TB/"
        DATA_PATH = BASE_PATH + "data/LC_Quad/"
        MODEL_PATH = BASE_PATH + "model_utils/models/LC_Quad/"
        GPT_MODEL_PATH = BASE_PATH + "model_utils/models/llm/"

        model_basic_config = {
                        "vocab_size": 50257,
                        "context_length": 1024,
                        "allowed_max_length": 1024,
                        "drop_rate": 0.1,
                        "qkv_bias": True,
                        "pad_token_id": 50256,
                        "ignore_index": -100
                    }

        self.model_config = {
            "gpt2-small (124M)":
                {
                    "name": "gpt2-small (124M)",
                    "model_size": "124M",
                    "models_dir": GPT_MODEL_PATH + "pre_train_gpt2_124M",
                    "basic_config": model_basic_config,
                    "model_config": {
                        "emb_dim": 768,
                        "n_layers": 12,
                        "n_heads": 12,
                    }
                },
            "gpt2-medium (355M)":
                {
                    "name": "gpt2-medium (355M)",
                    "model_size": "355M",
                    "models_dir": GPT_MODEL_PATH + "pre_train_gpt2_355M",
                    "basic_config": model_basic_config,
                    "model_config": {
                        "emb_dim": 1024,
                        "n_layers": 24,
                        "n_heads": 16,
                    }
                },
            "gpt2-large (774M)":
                {
                    "name": "gpt2-large (774M)",
                    "model_size": "774M",
                    "models_dir": GPT_MODEL_PATH + "pre_train_gpt2_774M",
                    "basic_config": model_basic_config,
                    "model_config": {
                        "emb_dim": 1280,
                        "n_layers": 36,
                        "n_heads": 20,
                    }
                },
            "gpt2-xl (1558M)":
                {
                    "name": "gpt2-large (1558M)",
                    "model_size": "1558M",
                    "models_dir": GPT_MODEL_PATH + "pre_train_gpt2_1558M",
                    "basic_config": model_basic_config,
                    "model_config": {
                        "emb_dim": 1600,
                        "n_layers": 48,
                        "n_heads": 25,
                    }
                },
        }

        self.lcquad_config = {
            "data":{
                    "train_data": DATA_PATH + "train.csv",
                    "test_data": DATA_PATH + "test.csv",
                    "sparql_wikidata_eids_labels_mapping": DATA_PATH + "sparql_wikidata_eid_label_map.json",
                    "sparql_wikidata_labels_eids_mapping": DATA_PATH + "sparql_wikidata_label_eid_map.json",
                    "modf_train_data": DATA_PATH + "modf_train_data.csv",
                    "modf_valid_data": DATA_PATH + "modf_valid_data.csv",
                    "modf_test_data": DATA_PATH + "modf_test_data.csv",
                    "train_dataset": DATA_PATH + "train_dataset.pt",
                    "val_dataset": DATA_PATH + "valid_dataset.pt",
                    "test_dataset": DATA_PATH + "test_dataset.pt",
                },
            "model": {
                "chosen_model": "gpt2-xl (1558M)",
                "tokenizer": "gpt2",
                "gpt_config": None,
                "device": None,
                "num_workers": 0,
                "batch_size": {
                    "train_batch_size": 8,
                    "test_batch_size": 4,
                    "val_batch_size": 4,
                },
                "model_path": MODEL_PATH + "lcquad_model_{model_ind}_" + str(timestamp_str) + ".pth",
                "inf_model_path": MODEL_PATH + "lcquad_model_{model_ind}.pth",
                "num_epochs": 3,
                "eval_freq": 500,
            }
        }
        return

    def get_config(self,):

        choosen_model = self.lcquad_config["model"]["chosen_model"]
        self.lcquad_config['model']["gpt_config"] = self.model_config[choosen_model]
        # self.lcquad_config['model']['device'] = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.lcquad_config['model']['device'] = "cpu"

        return self.lcquad_config