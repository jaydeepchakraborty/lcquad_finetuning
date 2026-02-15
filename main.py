import os
import sys

# os.environ["ACCELERATE_USE_CPU"] = "1"
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["ACCELERATE_DISABLE_MPS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from util.util_lib import *
import lcquad_finetuning.tokens.lcquad_tokens as lcquad_tokens
from lcquad_finetuning.config.lcquad_config import LCQuadConfig
from lcquad_finetuning.util.lcquad_logger import LCQuadLogger
from lcquad_finetuning.init.lcquad_init import LCQuadInit
from lcquad_finetuning.data.lcquad_datahelper import LCQUADDataHelper

from lcquad_finetuning.model.dapt.lcquad_dapt_modelhelper import LCQUADDAPTMODELHelper
from lcquad_finetuning.model.sft.lcquad_sft_modelhelper import LCQUADSFTMODELHelper
from lcquad_finetuning.model.rm.lcquad_rm_modelhelper import LCQUADRMMODELHelper
from lcquad_finetuning.model.rlhf.lcquad_rlhf_modelhelper import LCQUADRLHFMODELHelper
from lcquad_finetuning.inference_engine.lcquad_inf import LCQUADInfHelper
#
# from huggingface_hub import login
# login(token = lcquad_tokens.HUGGINGFACE_TOKEN)


def main():
    # Setting RANDOM SEED
    torch.manual_seed(123)
    np.random.seed(123)

    ########################################################
    # Step-0
    # loading config, loading logger
    ########################################################
    lcquad_conf_obj = LCQuadConfig()
    lcquad_conf = lcquad_conf_obj.load_config()

    lcquad_log_obj = LCQuadLogger(lcquad_conf)
    lcquad_log = lcquad_log_obj.get_logger()
    ########################################################

    ########################################################
    # Step-1
    # Loading the base data
    ########################################################
    # lcquad_init = LCQuadInit(lcquad_conf, lcquad_log)
    # lcquad_init.lcquad_init()
    ########################################################

    ########################################################
    # Step-2
    # preprocessing dataset (LCQUAD)
    ########################################################
    lcquaddata_helper = LCQUADDataHelper(lcquad_conf, lcquad_log)
    # lcquaddata_helper.preprocess_data()
    ########################################################

    ########################################################
    # STEP-3
    # DAPT (domain adaptive pretraining) LCQUAD model
    # training, testsing 
    ########################################################
    # # generate data for DAPT model train
    # lcquaddata_helper.populate_dapt_dataset()
    lcquad_dapt_model_helper = LCQUADDAPTMODELHelper(lcquad_conf, lcquad_log)
    # training the DAPT model
    dapt_trainer = lcquad_dapt_model_helper.training_lcquad_dapt_model()
    # saving the DAPT model
    lcquad_dapt_model_helper.save_lcquad_dapt_model(dapt_trainer)
    # # testing the DAPT model
    # lcquad_dapt_model_helper.test_lcquad_dapt_model()
    ########################################################


    ########################################################
    # STEP-4
    # SFT (Supervised Finetuning) LCQUAD model
    # training, testing instruction based finetuning
    #######################################################
    # generate data for SFT model train
    # lcquaddata_helper.populate_sft_dataset()
    # lcquad_sft_model_helper = LCQUADSFTMODELHelper(lcquad_conf, lcquad_log)
    # # training the SFT model
    # sft_trainer = lcquad_sft_model_helper.training_lcquad_sft_model()
    # # saving the SFT model
    # lcquad_sft_model_helper.save_lcquad_sft_model(sft_trainer)
    # # testing the SFT model
    # lcquad_sft_model_helper.test_lcquad_sft_model()
    # # inference the SFT model
    # lcquad_sft_model_helper.predict_top_K_lcquad_sft_model_helper()
    #######################################################


    ########################################################
    # STEP-5
    # RM (Reward Model) LCQUAD model
    # training, testing Reaward model (for feedback)
    #######################################################
    # # generate data for RL model train
    # lcquad_rm_model_helper = LCQUADRMMODELHelper(lcquad_conf, lcquad_log)
    # # generating reward model train, validation and test data
    # lcquad_rm_model_helper.generate_reward_data()
    # lcquaddata_helper.populate_rm_dataset()
    # # training the reward model
    # rm_model = lcquad_rm_model_helper.train_reward_model_helper()
    # # saving the reward model
    # lcquad_rm_model_helper.save_reward_model(rm_model)
    #######################################################


    ########################################################
    # STEP-6
    # RLHF (Reinforcement Learning with HumanFeedback) LCQUAD model
    # Reinforcement Learning PPO and update SFT model parameters
    #######################################################
    # # generate data for RLHF-PPO model train
    # lcquaddata_helper.populate_rlhf_dataset()
    # # training the RLHF-PPO model
    # lcquad_rlhf_model_helper = LCQUADRLHFMODELHelper(lcquad_conf, lcquad_log)
    # rlhf_model = lcquad_rlhf_model_helper.train_policy_model()
    # # saving the RLHF-PPO model
    # lcquad_rlhf_model_helper.save_policy_model(rlhf_model)
    #######################################################


    ########################################################
    # STEP-7
    # Inference LCQUAD model
    # on the final updated SFT model after RL (PPO) update
    #######################################################
    # generate LCQUAD test data
    # lcquaddata_helper.populate_lcquad_inf_dataset()
    # populating lcquad test result and scores
    # lcquad_inf_helper = LCQUADInfHelper(lcquad_conf, lcquad_log)
    # lcquad_inf_helper.lcquad_test()
    #######################################################

    return

"""
python3 main.py
"""
if __name__ == "__main__":
    try:
        strt_tm = time.perf_counter()
        main()
        end_tm = time.perf_counter()
        elapsed_tm = end_tm - strt_tm
        hr = int(elapsed_tm // 3600)
        min = int((elapsed_tm % 3600) // 60)
        sec = int(elapsed_tm % 60)
        print(f"Total time: {hr:02}:{min:02}:{sec:02}")
    except Exception as e:
        traceback.print_exc()