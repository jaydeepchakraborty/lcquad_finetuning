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
from lcquad_finetuning.util.lcquad_exception import LCQUADException
from lcquad_finetuning.util.util_lib import *

class LCQUADRMMODELHelper:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def load_policy_dataloder(self):
        lcquad_datahelper = LCQUADDataHelper(self.config, self.logger)
        dataloader = lcquad_datahelper.load_sft_dataloader()
        return dataloader

    def load_sft_model(self):
        model_path = self.config['model']['rlhf_model_path']
        self.logger.info(f"loading model from {model_path}")

        if self.config['model']['chosen_model'] == "gpt2":
            model_obj = GPT2LMHeadModel.from_pretrained(model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            model_obj = AutoModelForCausalLM.from_pretrained(model_path)
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        return model_obj


    def generate_reward_ip_dataset(self):

        return None

    def train_reward_model(self):

        train_dataloader = self.load_policy_dataloder()

        lcquad_sft_model = self.load_sft_model()

        lcquad_reward_dataset = self.generate_reward_ip_dataset()

        # for loop train

        # save reward model



