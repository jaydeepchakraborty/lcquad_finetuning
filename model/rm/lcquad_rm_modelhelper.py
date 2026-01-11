"""
Step 1: Load the instruction based trained SFT model

Step 2: load training data
["question", "sparql", "entity"]
"question":
What is the job of Stephane Mallarme, whose field of employment is translation?
"sparql":
SELECT ?answer WHERE { wd:Q767 wdt:P106 ?answer . ?answer wdt:P425 wd:Q7553}
"entity":
Question: What is the job of Stephane Mallarme, whose field of employment is translation?
<SPARQL> SELECT ?answer WHERE { wd:Q767 wdt:P106 ?answer . ?answer wdt:P425 wd:Q7553}

Step 3: Generate multiple outputs (top-K candidates) for each train sample using the SFT model

Step 4: Compare generated SPARQL vs reference and generate score
["question", "sparql", "entity", "generated sparql"]

Step 5: create reward-labeled dataset
["question", "sparql", "entity", "generated sparql", "score"]
Input example:
"entity":
Question: What is the job of Stephane Mallarme, whose field of employment is translation?
sparql: <here instead original SPARQL, use the generated SPARQL from the SFT model>

Step 6: train reward model
class RewardModel(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(base_model_name)
        self.head = nn.Linear(self.model.config.hidden_size, 1)  # scalar reward

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]  # last token
        reward = self.head(last_hidden)
        return reward

Step 7: Save the reward model
"""
from lcquad_finetuning.data.lcquad_datahelper import LCQUADDataHelper
from lcquad_finetuning.model.rm.lcquad_rm_model import LCQUADRMModel
from lcquad_finetuning.model.sft.lcquad_sft_modelhelper import LCQUADSFTMODELHelper
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer
from lcquad_finetuning.util.lcquad_exception import LCQUADException
from lcquad_finetuning.model.rm.lcquad_rm_reward_score import LCQUADRMRewardScoreGenerator
from lcquad_finetuning.util.lcquad_util import LCQuadUtil
from lcquad_finetuning.util.util_lib import *

class LCQUADRMMODELHelper:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def generate_reward_ip_dataset(self):
        # loading the trained SFT model
        lcquad_sft_model_helper = LCQUADSFTMODELHelper(self.config, self.logger)
        lcquad_sft_model_helper.predict_top_K_lcquad_sft_model_helper(padding_ind="left")

        # generate reward score
        lcquad_score_gen = LCQUADRMRewardScoreGenerator(self.config, self.logger)
        train_data_fl_path = self.config['data']['rm_train_data']
        train_data_df = pd.read_csv(train_data_fl_path)
        train_data_fl_path_with_reward = self.config['data']['rm_train_with_reward_score_data']
        lcquad_score_gen.generate_reward_score(train_data_df, train_data_fl_path_with_reward)

        test_data_fl_path = self.config['data']['rm_test_data']
        test_data_df = pd.read_csv(test_data_fl_path)
        test_data_fl_path_with_reward = self.config['data']['rm_test_with_reward_score_data']
        lcquad_score_gen.generate_reward_score(test_data_df, test_data_fl_path_with_reward)

        return None

    def load_tokenizer(self):
        lcquad_tokenizer_obj = LCQUADTokenizer(self.config, self.logger)
        tokenizer = lcquad_tokenizer_obj.load_tokenizer()
        return tokenizer

    def load_rm_dataloder(self):
        # loading the tokenizer
        tokenizer = self.load_tokenizer()

        lcquad_data_loader_obj = LCQUADDataHelper(self.config, self.logger)
        dataset_file_path = self.config['data']["rm_train_with_reward_score_dataset"]
        train_dataloader = lcquad_data_loader_obj.load_rm_dataloader(tokenizer, dataset_file_path, "train", "right")
        self.logger.info(f"train dataloader batches:- {len(train_dataloader)}")

        dataset_file_path = self.config['data']["rm_test_with_reward_score_dataset"]
        test_dataloader = lcquad_data_loader_obj.load_rm_dataloader(tokenizer, dataset_file_path, "test", "right")
        self.logger.info(f"test dataloader batches:- {len(test_dataloader)}")

        return train_dataloader, test_dataloader


    def load_lcquad_clm_model(self):

        model_path = self.config['model']['clm_model_path']
        self.logger.info(f"loading model from {model_path}")

        if self.config['model']['chosen_model'] == "gpt2":
            model_obj = GPT2LMHeadModel.from_pretrained(model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            model_obj = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, device_map=None)
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        return model_obj

    def train_rm_model(self, rm_model, train_rm_dataloader, tokenizer):

        effective_batch_size = self.config['model']['batch_size']['effective_batch_size']
        real_batch_size = self.config['model']['batch_size']['train_batch_size']
        accum_steps = effective_batch_size // real_batch_size

        optimizer = torch.optim.AdamW(rm_model.head.parameters(), lr=1e-5)
        loss_fn = torch.nn.MSELoss()

        rm_model.train()
        optimizer.zero_grad()

        num_epochs = self.config['model']['num_epochs']
        for epoch in range(num_epochs):
            total_loss = 0.0

            for batch_id, batch_data in enumerate(train_rm_dataloader):
                input_ids = batch_data['ip_padded_token_ids']
                rewards_gt = batch_data['reward_scores'].float()

                attention_mask = (input_ids != tokenizer.pad_token_id).long()

                rewards_pred = rm_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                raw_loss = loss_fn(rewards_pred, rewards_gt)
                loss = raw_loss / accum_steps
                loss.backward()

                total_loss += raw_loss.item()

                if (batch_id + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(rm_model.head.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            avg_loss = total_loss / len(train_rm_dataloader)
            self.logger.info(f"Epoch {epoch + 1} | RM loss: {avg_loss:.4f}")

        return rm_model

    def train_reward_model_helper(self):

        # loading the rm dataloader
        train_rm_dataloader, test_rm_dataloader = self.load_rm_dataloder()

        # loading tokenizer
        tokenizer = self.load_tokenizer()

        # loading the base Causal Language model (trained on new tokens)
        clm_model = self.load_lcquad_clm_model()
        for p in clm_model.parameters():
            p.requires_grad = False
        clm_model.eval()

        device = self.config["model"]["device"]
        rm_model_obj = LCQUADRMModel(clm_model, self.config, self.logger).to(device)
        rm_model = self.train_rm_model(rm_model_obj, train_rm_dataloader, tokenizer)

        return rm_model

    def save_reward_model(self, rm_model):
        # save reward model
        save_dir = self.config['model']['rm_model_path']
        rm_model.model.save_pretrained(save_dir)
        # save reward head
        torch.save(
            rm_model.head.state_dict(),
            f"{save_dir}/reward_head.pt"
        )
        self.logger.info(f"save reward model to {save_dir}")

        save_dir = save_dir.replace('latest', LCQuadUtil.get_curr_tm())
        rm_model.model.save_pretrained(save_dir)
        # save reward head
        torch.save(
            rm_model.head.state_dict(),
            f"{save_dir}/reward_head.pt"
        )
        self.logger.info(f"save reward model to {save_dir}")

        return
