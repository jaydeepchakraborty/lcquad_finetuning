from lcquad_finetuning.data.lcquad_datahelper import LCQUADDataHelper
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

    def load_policy_model(self):
        model_path = self.config['model']['sft_model_path']
        self.logger.info(f"loading model from {model_path}")

        if self.config['model']['chosen_model'] == "gpt2":
            model_obj = GPT2LMHeadModel.from_pretrained(model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            model_obj = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, device_map=None)
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        model_obj.eval()  # evaluation mode
        device = self.config['model']['device']
        model_obj.to(device)

        return model_obj

    def load_reward_model(self):

        save_dir = self.config['model']['rm_model_path']
        device = self.config['model']['device']
        base_model = AutoModelForCausalLM.from_pretrained(save_dir)

        rm_model = LCQUADRMModel(base_model, self.config, self.logger)
        rm_model.head.load_state_dict(
            torch.load(f"{save_dir}/reward_head.pt", map_location=device)
        )
        rm_model.to(device)

        rm_model.eval()  # evaluation mode
        return rm_model

    def load_ploicy_tokenizer(self):
        lcquad_tokenizer_obj = LCQUADTokenizer(self.config, self.logger)
        tokenizer = lcquad_tokenizer_obj.load_tokenizer()
        return tokenizer

    def load_policy_dataset(self):
        dataset_path = self.config['data']['rlhf_train_dataset']
        lcquad_datahelper = LCQUADDataHelper(self.config, self.logger)
        dataloader = lcquad_datahelper.load_rlhf_dataset(dataset_path)
        return dataloader

    def train_policy_model(self):

        # loading the policy train data
        policy_train_dataset = self.load_policy_dataset()

        # loading the policy tokenizer
        policy_tokenizer = self.load_ploicy_tokenizer()

        # -----------------------------
        # 1. Load SFT (policy) model
        # -----------------------------
        policy_model = self.load_policy_model()

        # -----------------------------
        # 2. Create frozen reference model
        # -----------------------------
        ref_model = copy.deepcopy(policy_model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        device = self.config['model']['device']
        ref_model.to(device)

        # -----------------------------
        # 3. Create reward model
        # -----------------------------
        reward_model = self.load_reward_model()

        # -----------------------------
        # 4. Create value model (baseline)
        # -----------------------------
        value_model = copy.deepcopy(policy_model)
        value_model.eval()
        for p in value_model.parameters():
            p.requires_grad = False
        device = self.config['model']['device']
        value_model.to(device)

        # train the PPO model
        lcquad_rlhf_model_obj = LCQUADRLHFModel(self.config, self.logger)
        lcquad_rlhf_model = lcquad_rlhf_model_obj.train_lcquad_rlhf_model(policy_tokenizer, policy_model, ref_model, reward_model, value_model, policy_train_dataset)

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




