from lcquad_finetuning.util.util_lib import *
import lcquad_finetuning.util.lcquad_cnst as lcquad_cnst
from lcquad_finetuning.util.lcquad_util import LCQuadUtil
from lcquad_finetuning.util.lcquad_exception import LCQUADException
from lcquad_finetuning.data.lcquad_datahelper import LCQUADDataHelper
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer
from lcquad_finetuning.model.sft.lcquad_sft_model import LCQUADSFTModel
from lcquad_finetuning.model.lcquad_modelhelper import LCQUADMODELHelper


class LCQUADSFTMODELHelper:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def load_tokenizer(self):
        lcquad_tokenizer_obj = LCQUADTokenizer(self.config, self.logger)
        tokenizer = lcquad_tokenizer_obj.load_tokenizer()
        return tokenizer

    def load_lcquad_dapt_model(self):
        lcquad_modelhelper = LCQUADMODELHelper(self.config, self.logger)
        model_obj = lcquad_modelhelper.load_model("lcquad_dapt_for_sft_model")
        return model_obj

    def save_lcquad_sft_model(self, lcquad_sft_model):
        model_path = self.config['model']['sft_model']['sft_model_path']

        if self.config['model']['chosen_model'] == lcquad_cnst.MODEL_GPT:
            # GPT2-XL: saves full model weights
            lcquad_sft_model.save_pretrained(model_path)
            self.logger.info(f"model saved to {model_path}")
            ts_model_path = model_path.replace("latest", LCQuadUtil.get_curr_tm())
            lcquad_sft_model.save_pretrained(ts_model_path)
            self.logger.info(f"model saved to {ts_model_path}")
        elif self.config['model']['chosen_model'] == lcquad_cnst.MODEL_QWEN:
            # Qwen: saves LoRA adapter weights only
            lcquad_sft_model.save_pretrained(model_path)
            self.logger.info(f"model saved to {model_path}")
            ts_model_path = model_path.replace("latest", LCQuadUtil.get_curr_tm())
            lcquad_sft_model.save_pretrained(ts_model_path)
            self.logger.info(f"model saved to {ts_model_path}")
        elif self.config['model']['chosen_model'] == lcquad_cnst.MODEL_MISTRAL:
            # Mistral: saves LoRA adapter weights only
            lcquad_sft_model.save_pretrained(model_path)
            self.logger.info(f"model saved to {model_path}")
            ts_model_path = model_path.replace("latest", LCQuadUtil.get_curr_tm())
            lcquad_sft_model.save_pretrained(ts_model_path)
            self.logger.info(f"model saved to {ts_model_path}")
        elif self.config['model']['chosen_model'] == lcquad_cnst.MODEL_GEMMA:
            # Gemma: saves LoRA adapter weights only
            lcquad_sft_model.save_pretrained(model_path)
            self.logger.info(f"model saved to {model_path}")
            ts_model_path = model_path.replace("latest", LCQuadUtil.get_curr_tm())
            lcquad_sft_model.save_pretrained(ts_model_path)
            self.logger.info(f"model saved to {ts_model_path}")
        elif self.config['model']['chosen_model'] == lcquad_cnst.MODEL_LLAMA:
            # Llama: saves LoRA adapter weights only
            lcquad_sft_model.save_pretrained(model_path)
            self.logger.info(f"model saved to {model_path}")
            ts_model_path = model_path.replace("latest", LCQuadUtil.get_curr_tm())
            lcquad_sft_model.save_pretrained(ts_model_path)
            self.logger.info(f"model saved to {ts_model_path}")
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        return

    def training_lcquad_sft_model(self, ):

        # loading the tokenizer
        tokenizer = self.load_tokenizer()

        lcquad_data_loader_obj = LCQUADDataHelper(self.config, self.logger)
        dataset_file_path = self.config['data']["sft_train_dataset"]
        train_dataloader = lcquad_data_loader_obj.load_sft_dataloader(tokenizer, dataset_file_path,
                                                                      "train", "right", "prompt_with_response")
        self.logger.info(f"train dataloader batches:- {len(train_dataloader)}")

        dataset_file_path = self.config['data']["sft_val_dataset"]
        val_dataloader = lcquad_data_loader_obj.load_sft_dataloader(tokenizer, dataset_file_path,
                                                                    "val", "right", "prompt_with_response")
        self.logger.info(f"val dataloader batches:- {len(val_dataloader)}")

        # training the LCQUAD model
        model = self.load_lcquad_dapt_model()
        device = self.config['model']['device']
        self.logger.info(f"device:- {device}")
        model.to(device)

        lcquad_model_sft_obj = LCQUADSFTModel(self.config, self.logger)
        lcquad_sft_model = lcquad_model_sft_obj.train_lcquad_sft_model(model, train_dataloader, val_dataloader)

        return lcquad_sft_model

    def load_lcquad_sft_model(self):
        lcquad_modelhelper = LCQUADMODELHelper(self.config, self.logger)
        model_obj = lcquad_modelhelper.load_model("lcquad_sft_model")
        return model_obj

    def test_lcquad_sft_model(self):
        # loading the tokenizer
        tokenizer = self.load_tokenizer()

        dataset_file_path = self.config['data']["sft_test_dataset"]
        lcquad_data_loader_obj = LCQUADDataHelper(self.config, self.logger)
        test_dataloader = lcquad_data_loader_obj.load_sft_dataloader(tokenizer, dataset_file_path,
                                                                     "test", "right", "prompt_with_response")
        self.logger.info(f"test dataloader {len(test_dataloader)}")

        lcquad_model_obj = LCQUADSFTModel(self.config, self.logger)
        lcquad_model = self.load_lcquad_sft_model()

        test_loss = lcquad_model_obj.calc_loss_loader(test_dataloader, lcquad_model)
        self.logger.info(f"test loss:- {test_loss:3f}")


    def predict_top_K_lcquad_sft_model(self, dataloader, tokenizer, model, k=1):

       # allowed_max_length = self.config['model']["model_config"]['basic_config']['allowed_max_length']
        allowed_max_length = 64

        generated_rows = []
        with torch.inference_mode(): # same no_grad
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["ip_modf_token_ids"]
                attention_mask = batch["attention_mask"]

                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=allowed_max_length,
                    do_sample=False,
                    num_beams=k, # how many outputs are generated
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

                gen_tokens = outputs[:, input_ids.shape[1]:]
                gen_texts = tokenizer.batch_decode(
                    gen_tokens,
                    skip_special_tokens=True
                )

                generated_rows.extend(
                    {
                        "prompt": e,
                        "question": q,
                        "original_sparql": s,
                        "generated_sparql": g.strip(),
                    }
                    for e, q, s, g in zip(
                        batch["org_txt"],
                        batch["question"],
                        batch["org_sparql"],
                        gen_texts
                    )
                )

                if batch_idx%500 == 0:
                    self.logger.info(f"output generation is done for batch_idx: {batch_idx}")

        generated_df = pd.DataFrame(generated_rows)

        return generated_df

    def predict_top_K_lcquad_sft_model_helper(self):

        padding_ind = "right"

        # loading the tokenizer
        tokenizer = self.load_tokenizer()
        tokenizer.padding_side = padding_ind # during inference default padding is right
        # loading the trained SFT model
        lcquad_model = self.load_lcquad_sft_model()
        lcquad_model.config.use_cache = True # enable KV caching during inference
        lcquad_model.eval()

        lcquad_data_loader_obj = LCQUADDataHelper(self.config, self.logger)

        # generating training output
        train_dataset_file_path = self.config['data']["sft_train_dataset"]
        """
        passing as "test", to use "customized_test_right_pad_collate_fn()"
        """
        train_dataloader = lcquad_data_loader_obj.load_sft_dataloader(tokenizer, train_dataset_file_path,
                                                                      "test", padding_ind, "prompt_without_response")
        self.logger.info(f"train dataloader {len(train_dataloader)}")
        train_generated_df = self.predict_top_K_lcquad_sft_model(train_dataloader, tokenizer, lcquad_model)
        sft_train_result_datapath = self.config['data']["sft_train_result_data"]
        train_generated_df.to_csv(sft_train_result_datapath, index=False)
        self.logger.info(f"SFT output(train) {sft_train_result_datapath}")

        # generating validation output
        valid_dataset_file_path = self.config['data']["sft_val_dataset"]
        """
        passing as "test", to use "customized_test_right_pad_collate_fn()"
        """
        valid_dataloader = lcquad_data_loader_obj.load_sft_dataloader(tokenizer, valid_dataset_file_path,
                                                                      "test", padding_ind, "prompt_without_response")
        self.logger.info(f"valid dataloader {len(valid_dataloader)}")
        valid_generated_df = self.predict_top_K_lcquad_sft_model(valid_dataloader, tokenizer, lcquad_model)
        sft_valid_result_datapath = self.config['data']["sft_val_result_data"]
        valid_generated_df.to_csv(sft_valid_result_datapath, index=False)
        self.logger.info(f"SFT output(valid) {sft_valid_result_datapath}")

        # generating test output
        test_dataset_file_path = self.config['data']["sft_test_dataset"]
        test_dataloader = lcquad_data_loader_obj.load_sft_dataloader(tokenizer, test_dataset_file_path,
                                                                     "test", padding_ind, "prompt_without_response")
        self.logger.info(f"test dataloader {len(test_dataloader)}")
        test_generated_df = self.predict_top_K_lcquad_sft_model(test_dataloader, tokenizer, lcquad_model)
        sft_test_result_datapath = self.config['data']["sft_test_result_data"]
        test_generated_df.to_csv(sft_test_result_datapath, index=False)
        self.logger.info(f"SFT output(test) {sft_test_result_datapath}")


