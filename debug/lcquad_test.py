import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.config.lcquad_config import LCQuadConfig
from lcquad_finetuning.data.lcquad_datahelper import LCQUADDataHelper
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer
from lcquad_finetuning.util.lcquad_logger import LCQuadLogger
from lcquad_finetuning.data.lcquad_dapt_dataset import LCQUADDAPTDataset
from lcquad_finetuning.data.lcquad_sft_dataset import LCQUADSFTDataset
from lcquad_finetuning.model.lcquad_modelhelper import LCQUADMODELHelper


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

    # #load tokenizer
    # lcquad_tokenizer_obj = LCQUADTokenizer(config, logger)
    # tokenizer = lcquad_tokenizer_obj.load_tokenizer()
    #
    # # load train-dataloader
    # lcquad_data_loader_obj = LCQUADDataHelper(config, logger)
    # dataset_file_path = config['data']["sft_train_dataset"]
    # train_dataloader = lcquad_data_loader_obj.load_sft_dataloader(tokenizer, dataset_file_path, "train")
    # print(f"train dataloader batches:- {len(train_dataloader)}")
    #
    # for batch_id, batch_data in enumerate(train_dataloader):
    #     print("org_txt:- ")
    #     print(batch_data['org_txt'])
    #     print("ip_org_text_lst:- ")
    #     print(batch_data['ip_org_text_lst'])
    #     print("ip_org_token_ids:- ")
    #     print(batch_data['ip_org_token_ids'])
    #     print("ip_modf_text_lst:- ")
    #     print(batch_data['ip_modf_text_lst'])
    #     print("ip_modf_token_ids:- ")
    #     print(batch_data['ip_modf_token_ids'])
    #     print("lbl_org_text_lst:- ")
    #     print(batch_data['lbl_org_text_lst'])
    #     print("lbl_org_token_ids:- ")
    #     print(batch_data['lbl_org_token_ids'])
    #     print("lbl_modf_text_lst:- ")
    #     print(batch_data['lbl_modf_text_lst'])
    #     print("lbl_modf_token_ids:- ")
    #     print(batch_data['lbl_modf_token_ids'])
    #     break

    padding_ind = "right"

    # loading the tokenizer
    lcquad_tokenizer_obj = LCQUADTokenizer(config, logger)
    tokenizer = lcquad_tokenizer_obj.load_tokenizer()
    tokenizer.padding_side = padding_ind  # during inference default padding is right

    lcquad_data_loader_obj = LCQUADDataHelper(config, logger)
    # generating training output
    train_dataset_file_path = config['data']["sft_train_dataset"]
    """
    passing as "test", to use "customized_test_right_pad_collate_fn()"
    """
    train_dataloader = lcquad_data_loader_obj.load_sft_dataloader(tokenizer, train_dataset_file_path,
                                                                  "test", padding_ind, "prompt_without_response")
    for batch_data in train_dataloader:
        print(batch_data)
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

    print(f"BEFORE MODIFIED:")
    print(f"PAD TOKEN: {tokenizer.pad_token}")
    txt_arr = ['<PAD>', '<Q_START>', '<Q_END>', '<|endoftext|>', '<SPARQL_START>', '<SPARQL_END>']
    for txt_tok in txt_arr:
        token_id = tokenizer.convert_tokens_to_ids(txt_tok)
        print(f"TEXT TOKEN: {txt_tok}\t TOKEN ID: {token_id}")

    # from path
    tokenizer_path = config["model"]["tokenizer_path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(tokenizer.vocab_size)
    print(len(tokenizer))

    print(f"AFTER MODIFIED:")
    print(f"PAD TOKEN: {tokenizer.pad_token}")
    txt_arr = ['<PAD>', '<Q_START>', '<Q_END>', '<|endoftext|>', '<SPARQL_START>', '<SPARQL_END>']
    for txt_tok in txt_arr:
        token_id = tokenizer.convert_tokens_to_ids(txt_tok)
        print(f"TEXT TOKEN: {txt_tok}\t TOKEN ID: {token_id}")

    newline_id = tokenizer.encode('\n', add_special_tokens=False)
    print(f"TEXT TOKEN: new_line\t TOKEN ID: {newline_id}")
    space_id = tokenizer.encode(' ', add_special_tokens=False)
    print(f"TEXT TOKEN: space\t TOKEN ID: {space_id}")

    # token_ids = [151665, 151666, 151664, 151663, 151662, 151661]
    # for token_id in token_ids:
    #     token_txt = tokenizer.convert_ids_to_tokens(token_id)
    #     print(f"TOKEN ID: {token_id}\t TEXT TOKEN: {token_txt}\t ")

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

def analysis_models(lcquad_conf, lcquad_log):
    model_path = lcquad_conf['model']['base_model_path']
    model_path = "/Volumes/Jay_4TB/model_utils/models/LC_Quad/lcquad_clm_model_Qwen/Qwen2.5-1.5B/latest"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total: {total_params / 1e6:.1f}M | Trainable: {trainable_params / 1e6:.1f}M")

def sft_model_training_test(config, logger):

    padding_ind = "right"

    # loading the tokenizer
    lcquad_tokenizer_obj = LCQUADTokenizer(config, logger)
    tokenizer = lcquad_tokenizer_obj.load_tokenizer()
    tokenizer.padding_side = padding_ind  # during inference default padding is left

    lcquad_data_loader_obj = LCQUADDataHelper(config, logger)
    # generating training output
    train_dataset_file_path = config['data']["sft_train_dataset"]
    """
    passing as "test", to use "customized_train_right_pad_collate_fn()"
    """
    train_dataloader = lcquad_data_loader_obj.load_sft_dataloader(tokenizer, train_dataset_file_path,
                                                                  "train", padding_ind, "prompt_with_response")

    # for batch in train_dataloader:
    #     print('org_txt')
    #     print(batch['org_txt'])
    #     print('ip_modf_token_ids')
    #     print(batch['ip_modf_token_ids'])
    #     print('attention_mask')
    #     print(batch['attention_mask'])
    #     print('lbl_modf_token_ids')
    #     print(batch['lbl_modf_token_ids'])
    #     break

    lcquad_modelhelper = LCQUADMODELHelper(config, logger)
    model = lcquad_modelhelper.load_model("lcquad_clm_for_sft_model")
    model.to("mps")
    model.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-5,
        betas=(0.9, 0.95),
        weight_decay=0.0
    )
    optimizer.zero_grad()

    accumulation_steps = 4

    for epoch in range(10):
        for batch_id, batch_data in enumerate(train_dataloader):
            input_batch = batch_data['ip_modf_token_ids']
            attention_mask_batch = batch_data['attention_mask']
            target_batch = batch_data['lbl_modf_token_ids']

            outputs = model(input_ids=input_batch, attention_mask=attention_mask_batch, labels=target_batch)
            loss = outputs.loss/ accumulation_steps
            loss.backward()
            print(f"epoch: {epoch} batch_id: {batch_id} loss: {loss.item()}")

            if (batch_id + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if batch_id == 16:
                break



def sft_model_inference_test(config, logger):

    padding_ind = "right"

    # loading the tokenizer
    lcquad_tokenizer_obj = LCQUADTokenizer(config, logger)
    tokenizer = lcquad_tokenizer_obj.load_tokenizer()
    tokenizer.padding_side = padding_ind  # during inference default padding is left

    lcquad_data_loader_obj = LCQUADDataHelper(config, logger)
    # generating training output
    train_dataset_file_path = config['data']["sft_train_dataset"]
    """
    passing as "test", to use "customized_test_right_pad_collate_fn()"
    """
    train_dataloader = lcquad_data_loader_obj.load_sft_dataloader(tokenizer, train_dataset_file_path,
                                                                  "test", padding_ind, "prompt_without_response")

    # loading the trained SFT model
    lcquad_modelhelper = LCQUADMODELHelper(config, logger)
    lcquad_model = lcquad_modelhelper.load_model("lcquad_sft_model")
    lcquad_model.config.use_cache = True  # enable KV caching during inference
    lcquad_model.eval()

    with torch.no_grad():
        for batch in train_dataloader:
            org_txt =  batch["org_txt"]
            input_ids = batch["ip_modf_token_ids"]
            attention_mask = batch["attention_mask"]

            for _ in range(64):
                logits = lcquad_model(
                    input_ids=batch['ip_modf_token_ids'],
                    attention_mask=batch['attention_mask']
                ).logits[:, -1, :]  # Get last token logits

                next_token = logits.argmax(dim=-1, keepdim=True)

                # Append new token
                batch['ip_modf_token_ids'] = torch.cat([batch['ip_modf_token_ids'], next_token], dim=1)

                # Update attention mask
                batch['attention_mask'] = torch.cat([
                    batch['attention_mask'],
                    torch.ones_like(next_token)
                ], dim=1)

            break

        # After generation loop, slice last 64 tokens from each sample
        generated_tokens = batch['ip_modf_token_ids'][:, -64:]

        # Decode to text
        outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        print("org_txt")
        print(org_txt)
        print("input_ids")
        print(input_ids)
        print("attention_mask")
        print(attention_mask)
        print("output tokens")
        print(generated_tokens)
        print("outputs")
        print(outputs)

    # allowed_max_length = 64
    #
    # generated_rows = []
    # with torch.inference_mode():  # same no_grad
    #     for batch_idx, batch in enumerate(train_dataloader):
    #         input_ids = batch["ip_modf_token_ids"]
    #         attention_mask = batch["attention_mask"]
    #
    #         outputs = lcquad_model.generate(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             max_new_tokens=allowed_max_length,
    #             do_sample=False,
    #             num_beams=1,  # how many outputs are generated
    #             pad_token_id=tokenizer.pad_token_id,
    #             eos_token_id=tokenizer.eos_token_id
    #         )
    #
    #         print("input_ids")
    #         print(input_ids)
    #         print("attention_mask")
    #         print(attention_mask)
    #         print("outputs")
    #         print(outputs)
    #
    #         gen_tokens = outputs[:, input_ids.shape[1]:]
    #         gen_texts = tokenizer.batch_decode(
    #             gen_tokens,
    #             skip_special_tokens=True
    #         )
    #
    #         generated_rows.extend(
    #             {
    #                 "prompt": e,
    #                 "question": q,
    #                 "original_sparql": s,
    #                 "generated_sparql": g.strip(),
    #             }
    #             for e, q, s, g in zip(
    #                 batch["org_txt"],
    #                 batch["question"],
    #                 batch["org_sparql"],
    #                 gen_texts
    #             )
    #         )
    #         print(generated_rows)
    #         break

    return

def chk_token_length(lcquad_conf, lcquad_log):
    # load all data
    train_df = pd.read_csv("/Volumes/Jay_4TB/data/LC_Quad/modf_train_data.csv")
    valid_df = pd.read_csv("/Volumes/Jay_4TB/data/LC_Quad/modf_valid_data.csv")
    test_df = pd.read_csv("/Volumes/Jay_4TB/data/LC_Quad/modf_test_data.csv")
    df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    #sparql
    #prompt_with_response

    sparqls = df["prompt_with_response"].unique()

    # load tokenizer
    tok_obj = LCQUADTokenizer(lcquad_conf, lcquad_log)
    tokenizer = tok_obj.load_tokenizer()

    # count token lengths
    lengths = [len(tokenizer(s)["input_ids"]) for s in sparqls]

    print(f"Total unique SPARQL: {len(lengths)}")
    print(f"Max token length: {max(lengths)}")
    print(f"Mean token length: {sum(lengths) / len(lengths):.1f}")
    print(f"Median token length: {sorted(lengths)[len(lengths) // 2]}")
    print(f"Samples > 128 tokens: {sum(1 for l in lengths if l > 128)}")
    print(f"95th percentile: {sorted(lengths)[int(len(lengths) * 0.95)]}")


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

    # analysis_models(lcquad_conf, lcquad_log)

    # sft_model_inference_test(lcquad_conf, lcquad_log)

    # sft_model_training_test(lcquad_conf, lcquad_log)

    chk_token_length(lcquad_conf, lcquad_log)
