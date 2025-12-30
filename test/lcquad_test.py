import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.config.lcquad_config import LCQuadConfig
from lcquad_finetuning.data.lcquad_datahelper import LCQUADDataHelper
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer
from lcquad_finetuning.util.lcquad_logger import LCQuadLogger


def chk_tokenizer(conf, logger):
    lcquad_tokenizer_obj = LCQUADTokenizer(conf, logger)
    tokenizer = lcquad_tokenizer_obj.load_tokenizer()
    # labels = [25, 6530, 262, 256, 2455, 560, 22765, 319, 13546, 9281, 5780, 13, 198, 50257, 198, 46506, 5633, 41484, 33411, 1391, 220, 50493, 220, 69607, 5633,
    #           55, 764, 5633, 55, 220, 63878, 5633, 41484, 92, 50256, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258,
    #           50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258]

    labels = [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 46506, 5633, 41484, 33411, 1391, 220, 50493, 220, 69607,
              5633, 55, 764, 5633, 55, 220, 63878, 5633, 41484, 92, 50256, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
              -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
    # lbl_org_text = tokenizer.decode(labels, skip_special_tokens=False)

    lbl_org_text = lcquad_tokenizer_obj.lcquad_tok_decoder(labels, tokenizer)
    print(lbl_org_text)

def print_dataloader(config, logger):

    #load tokenizer
    lcquad_tokenizer_obj = LCQUADTokenizer(config, logger)
    tokenizer = lcquad_tokenizer_obj.load_tokenizer()

    # load train-dataloader
    lcquad_data_loader_obj = LCQUADDataHelper(config, logger)
    dataset_file_path = config['data']["train_dataset"]
    train_dataloader = lcquad_data_loader_obj.load_dataloader(tokenizer, dataset_file_path, "train")
    print(f"train dataloader batches:- {len(train_dataloader)}")

    for batch_id, batch_data in enumerate(train_dataloader):
        print("org_txt:- ")
        print(batch_data['org_txt'])
        print("ip_org_text_lst:- ")
        print(batch_data['ip_org_text_lst'])
        print("ip_org_token_ids:- ")
        print(batch_data['ip_org_token_ids'])
        print("ip_modf_text_lst:- ")
        print(batch_data['ip_modf_text_lst'])
        print("ip_modf_token_ids:- ")
        print(batch_data['ip_modf_token_ids'])
        print("lbl_org_text_lst:- ")
        print(batch_data['lbl_org_text_lst'])
        print("lbl_org_token_ids:- ")
        print(batch_data['lbl_org_token_ids'])
        print("lbl_modf_text_lst:- ")
        print(batch_data['lbl_modf_text_lst'])
        print("lbl_modf_token_ids:- ")
        print(batch_data['lbl_modf_token_ids'])
        break


if __name__ == "__main__":
    # Setting RANDOM SEED
    torch.manual_seed(123)
    np.random.seed(123)

    # Step-0 loading config
    lcquad_conf_obj = LCQuadConfig()
    lcquad_conf = lcquad_conf_obj.load_config()

    lcquad_log_obj = LCQuadLogger(lcquad_conf)
    lcquad_log = lcquad_log_obj.get_logger()

    print_dataloader(lcquad_conf, lcquad_log)

    # chk_tokenizer(lcquad_conf, lcquad_log)
