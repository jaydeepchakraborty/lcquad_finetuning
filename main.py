import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from util.util_lib import *
from lcquad_finetuning.config.lcquad_config import LCQuadConfig
from lcquad_finetuning.util.lcquad_logger import LCQuadLogger
from lcquad_finetuning.data.lcquad_datahelper import LCQUADDataHelper
from lcquad_finetuning.model.clm.lcquad_clm_modelhelper import LCQUADCLMMODELHelper
from lcquad_finetuning.model.sft.lcquad_sft_modelhelper import LCQUADSFTMODELHelper


def main():
    # Setting RANDOM SEED
    torch.manual_seed(123)
    np.random.seed(123)

    ########################################################
    # Step-0
    # loading config, loading logger, initilize lcquad
    ########################################################
    lcquad_conf_obj = LCQuadConfig()
    lcquad_conf = lcquad_conf_obj.load_config()

    lcquad_log_obj = LCQuadLogger(lcquad_conf)
    lcquad_log = lcquad_log_obj.get_logger()

    # lcquad_init = LCQuadInit(lcquad_conf, lcquad_log)
    # lcquad_init.lcquad_init()
    ########################################################


    ########################################################
    # Step-1
    # preparing dataset (LCQUAD)
    ########################################################
    # lcquaddata_helper = LCQUADDataHelper(lcquad_conf, lcquad_log)
    # lcquaddata_helper.preprocess_data()
    # lcquaddata_helper.populate_clm_dataset()
    # lcquaddata_helper.populate_dataset()
    ########################################################


    ########################################################
    # STEP-2
    # CLM (Causal Language Model) LCQUAD model
    # domain adaptive pretraining
    ########################################################
    # lcquad_clm_model_helper = LCQUADCLMMODELHelper(lcquad_conf, lcquad_log)
    # lcquad_clm_model_helper.training_lcquad_clm_model()
    ########################################################


    ########################################################
    # STEP-3
    # training, testing, inference LCQUAD model (SFT)
    #######################################################
    lcquad_sft_model_helper = LCQUADSFTMODELHelper(lcquad_conf, lcquad_log)

    # training model
    lcquad_sft_model_helper.training_lcquad_sft_model()

    # test on trained model (LCQUAD)
    # lcquadmodel_helper.test_lcquad_model()
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