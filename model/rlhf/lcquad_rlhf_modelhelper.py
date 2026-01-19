from lcquad_finetuning.data.lcquad_datahelper import LCQUADDataHelper
from lcquad_finetuning.model.lcquad_modelhelper import LCQUADMODELHelper
from lcquad_finetuning.model.rlhf.lcquad_rlhf_model import LCQUADRLHFModel
from lcquad_finetuning.model.rm.lcquad_rm_model import LCQUADRMModel
from lcquad_finetuning.model.rm.lcquad_rm_modelhelper import LCQUADRMMODELHelper
from lcquad_finetuning.util.lcquad_exception import LCQUADException
from lcquad_finetuning.util.lcquad_util import LCQuadUtil
from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer

class LCQUADRLHFMODELHelper:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def load_policy_tokenizer(self):
        lcquad_tokenizer_obj = LCQUADTokenizer(self.config, self.logger)
        tokenizer = lcquad_tokenizer_obj.load_tokenizer()
        tokenizer.padding_side = "left"  # during inference default padding is left
        return tokenizer

    def load_policy_dataloader(self, tokenizer):
        lcquad_data_loader_obj = LCQUADDataHelper(self.config, self.logger)
        dataset_file_path = self.config['data']["rlhf_train_dataset"]
        dataloader = lcquad_data_loader_obj.load_rlhf_dataloader(tokenizer, dataset_file_path, "train", "left")
        self.logger.info(f"train dataloader batches:- {len(dataloader)}")
        return dataloader

    def load_policy_model(self):
        lcquad_modelhelper = LCQUADMODELHelper(self.config, self.logger)
        model_obj = lcquad_modelhelper.load_model("lcquad_policy_model")
        return model_obj

    def load_reference_model(self):
        lcquad_modelhelper = LCQUADMODELHelper(self.config, self.logger)
        model_obj = lcquad_modelhelper.load_model("lcquad_reference_model")
        return model_obj

    def load_reward_model(self):
        lcquad_modelhelper = LCQUADMODELHelper(self.config, self.logger)
        model_obj = lcquad_modelhelper.load_model("lcquad_reward_model")
        return model_obj

    def load_value_model(self):
        lcquad_modelhelper = LCQUADMODELHelper(self.config, self.logger)
        model_obj = lcquad_modelhelper.load_model("lcquad_value_model")
        return model_obj

    def train_policy_model(self):

        # loading the policy train data
        policy_tokenizer = self.load_policy_tokenizer()
        policy_train_dataloader = self.load_policy_dataloader(policy_tokenizer)

        # -----------------------------
        # 1. Load SFT (policy) model
        # -----------------------------
        policy_model = self.load_policy_model()

        # -----------------------------
        # 2. Load frozen reference model
        # -----------------------------
        reference_model = self.load_reference_model()

        # -----------------------------
        # 3. Load reward model
        # -----------------------------
        reward_model = self.load_reward_model()

        # -----------------------------
        # 4. Load value model
        # -----------------------------
        value_model = self.load_value_model()

        # train the PPO model
        lcquad_rlhf_model_obj = LCQUADRLHFModel(self.config, self.logger)
        lcquad_rlhf_model = lcquad_rlhf_model_obj.train_lcquad_rlhf_model(policy_tokenizer,
                                                                          policy_train_dataloader,
                                                                          policy_model,
                                                                          reference_model,
                                                                          reward_model,
                                                                          value_model)

        return lcquad_rlhf_model

    # saving the LCQUAD-RLHF model
    def save_policy_model(self, lcquad_rlhf_model):

        rlhf_model_path = self.config['model']['rlhf_model_path']
        if self.config['model']['chosen_model'] == "gpt2":
            lcquad_rlhf_model.save_pretrained(rlhf_model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            lcquad_rlhf_model.save_pretrained(rlhf_model_path)
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)
        self.logger.info(f"model saved to {rlhf_model_path}")

        rlhf_model_path = self.config['model']['rlhf_model_path']
        rlhf_model_path = rlhf_model_path.replace("latest", LCQuadUtil.get_curr_tm())
        if self.config['model']['chosen_model'] == "gpt2":
            lcquad_rlhf_model.save_pretrained(rlhf_model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            lcquad_rlhf_model.save_pretrained(rlhf_model_path)
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)
        self.logger.info(f"model saved to {rlhf_model_path}")

        inf_model_path = self.config['model']['inf_model_path']
        if self.config['model']['chosen_model'] == "gpt2":
            lcquad_rlhf_model.save_pretrained(inf_model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            lcquad_rlhf_model.save_pretrained(inf_model_path)
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)
        self.logger.info(f"model saved to {inf_model_path}")

        inf_model_path = self.config['model']['inf_model_path']
        inf_model_path = inf_model_path.replace("latest", LCQuadUtil.get_curr_tm())
        if self.config['model']['chosen_model'] == "gpt2":
            lcquad_rlhf_model.save_pretrained(inf_model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            lcquad_rlhf_model.save_pretrained(inf_model_path)
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)
        self.logger.info(f"model saved to {inf_model_path}")




