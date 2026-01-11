import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.config.lcquad_config import LCQuadConfig
from lcquad_finetuning.data.lcquad_datahelper import LCQUADDataHelper
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer
from lcquad_finetuning.util.lcquad_logger import LCQuadLogger
from lcquad_finetuning.data.lcquad_clm_dataset import LCQUADCLMDataset
from lcquad_finetuning.data.lcquad_sft_dataset import LCQUADSFTDataset


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
    dataset_file_path = config['data']["sft_train_dataset"]
    train_dataloader = lcquad_data_loader_obj.load_sft_dataloader(tokenizer, dataset_file_path, "train")
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

def analysis_dataframe(config, logger):

    fl_path = config['data']["base_train_data"]
    fl_df = pd.read_csv(fl_path)
    print(f"len:- {len(fl_df)} , shape:- {fl_df.shape}")
    print(fl_df.columns.tolist())
    print(fl_df.head())

    # 1. Split each string into a list of words
    # The .str accessor provides string methods.
    # split() without arguments splits by whitespace.
    word_lists = fl_df['sparql'].str.split()

    # 2. Get the length of each list (the word count for each row)
    # The .str accessor works on the series of lists to get their lengths.
    word_counts = word_lists.str.len()

    # 3. Find the maximum word count
    max_words = word_counts.max()

    print(f"The maximum number of words in the column is: {max_words}")
    # Output: The maximum number of words in the column is: 8

def analysis_json(config, logger):

    with open(config['data']['lcquad_token'], "r") as f:
        new_tokens = json.load(f)

    print(len(new_tokens))

def analysis_tokenizer(config, logger):
    # base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['tokenizer'])
    print(tokenizer.vocab_size)
    print(len(tokenizer))

    # from path
    tokenizer_path = config["model"]["tokenizer_path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(tokenizer.vocab_size)
    print(len(tokenizer))

def analysis_dataset(config, logger):
    # LCQUADCLMDataset
    # LCQUADSFTDataset
    dataset_file_path = config['data']['sft_test_dataset']
    logger.info(f"loading dataset from:- {dataset_file_path}")
    with torch.serialization.safe_globals([LCQUADSFTDataset]):
        dataset = torch.load(dataset_file_path, weights_only=False)

    print(f"dataset length: {len(dataset)}")

def analysis_dataloader(config, logger):

    # from path
    tokenizer_path = config["model"]["tokenizer_path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(tokenizer.vocab_size)
    print(len(tokenizer))

    lcquad_data_loader_obj = LCQUADDataHelper(config, logger)

    dataset_file_path = config['data']["sft_test_dataset"]
    dataloader = lcquad_data_loader_obj.load_sft_dataloader(tokenizer, dataset_file_path, "test")
    logger.info(f"dataloader {len(dataloader)}")


if __name__ == "__main__":
    # Setting RANDOM SEED
    torch.manual_seed(123)
    np.random.seed(123)

    # Step-0 loading config
    lcquad_conf_obj = LCQuadConfig()
    lcquad_conf = lcquad_conf_obj.load_config()

    lcquad_log_obj = LCQuadLogger(lcquad_conf)
    lcquad_log = lcquad_log_obj.get_logger()

    import trl
    print(trl.__version__)

    # analysis_dataframe(lcquad_conf, lcquad_log)

    # analysis_json(lcquad_conf, lcquad_log)

    # print_dataloader(lcquad_conf, lcquad_log)

    # chk_tokenizer(lcquad_conf, lcquad_log)

    # analysis_tokenizer(lcquad_conf, lcquad_log)

    # analysis_dataset(lcquad_conf, lcquad_log)

    # analysis_dataloader(lcquad_conf, lcquad_log)
