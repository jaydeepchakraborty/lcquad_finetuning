from lcquad_finetuning.util.lcquad_util import LCQuadUtil
from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.data.lcquad_datahelper import LCQUADDataHelper
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer
from lcquad_finetuning.model.sft.lcquad_sft_model import LCQUADSFTModel
from lcquad_finetuning.util.lcquad_exception import LCQUADException


class LCQUADSFTMODELHelper:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def load_tokenizer(self):
        lcquad_tokenizer_obj = LCQUADTokenizer(self.config, self.logger)
        tokenizer = lcquad_tokenizer_obj.load_tokenizer()
        return tokenizer

    def load_lcquad_clm_model(self):

        model_path = self.config['model']['clm_model_path']
        self.logger.info(f"loading model from {model_path}")

        if self.config['model']['chosen_model'] == "gpt2":
            model_obj = GPT2LMHeadModel.from_pretrained(model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            # for full supervised finetuning
            # model_obj = AutoModelForCausalLM.from_pretrained(model_path)
            # for QLoRA supervised finetuning
            """
            Apple MPS does not support 4-bit / 8-bit quantization
            So we are doing LoRA + fp16 for mac
            """
            # # Load model in 4-bit
            # bnb_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_compute_dtype=torch.bfloat16,
            #     bnb_4bit_use_double_quant=True
            # )
            # model_obj = AutoModelForCausalLM.from_pretrained(model_path,
            #                                                  quantization_config=bnb_config,)
            # Prepare for k-bit training
            # model_obj = prepare_model_for_kbit_training(model_obj)
            model_obj = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16,
                device_map=None
            )

            # Attach LoRA adapters
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen attention proj layers
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model_obj = get_peft_model(model_obj, lora_config)
            for name, param in model_obj.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
            model_obj.print_trainable_parameters()
            model_obj.enable_input_require_grads()
            # Enable gradient checkpointing
            model_obj.config.use_cache = False  # Required for checkpointing, On CPU or small GPUs
            model_obj.gradient_checkpointing_enable() # save memory, only if OOM ( Out of Memory )
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        return model_obj

    def save_lcquad_sft_model(self, lcquad_sft_model):
        model_path = self.config['model']['sft_model_path']

        if self.config['model']['chosen_model'] == "gpt2":
            lcquad_sft_model.save_pretrained(model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            lcquad_sft_model.save_pretrained(model_path)
            self.logger.info(f"model saved to {model_path}")
            model_path = model_path.replace("latest", LCQuadUtil.get_curr_tm())
            lcquad_sft_model.save_pretrained(model_path)
            self.logger.info(f"model saved to {model_path}")
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        return

    def load_lcquad_sft_model(self):

        if self.config['model']['chosen_model'] == "gpt2":
            sft_model_path = self.config['model']['sft_model_path']
            model_obj = GPT2LMHeadModel.from_pretrained(sft_model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            clm_model_path = self.config['model']['clm_model_path']
            self.logger.info(f"CLM model loaded from {clm_model_path}")
            sft_model_obj = AutoModelForCausalLM.from_pretrained(clm_model_path,
                                                             dtype=torch.float16,
                                                             device_map=None)
            device = self.config['model']['device']
            self.logger.info(f"device:- {device}")
            sft_model_obj.to(device)
            sft_model_path = self.config['model']['sft_model_path']
            self.logger.info(f"SFT model loaded from {sft_model_path}")
            model_obj = PeftModel.from_pretrained(
                sft_model_obj,
                sft_model_path
            )
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        return model_obj

    def training_lcquad_sft_model(self, ):

        # loading the tokenizer
        tokenizer = self.load_tokenizer()

        lcquad_data_loader_obj = LCQUADDataHelper(self.config, self.logger)
        dataset_file_path = self.config['data']["sft_train_dataset"]
        train_dataloader = lcquad_data_loader_obj.load_sft_dataloader(tokenizer, dataset_file_path, "train", "right")
        self.logger.info(f"train dataloader batches:- {len(train_dataloader)}")

        dataset_file_path = self.config['data']["sft_val_dataset"]
        val_dataloader = lcquad_data_loader_obj.load_sft_dataloader(tokenizer, dataset_file_path, "val", "right")
        self.logger.info(f"val dataloader batches:- {len(val_dataloader)}")

        # training the LCQUAD model
        model = self.load_lcquad_clm_model()
        lcquad_model_sft_obj = LCQUADSFTModel(self.config, self.logger)
        lcquad_sft_model = lcquad_model_sft_obj.train_lcquad_sft_model(model, train_dataloader, val_dataloader)

        return lcquad_sft_model

    def test_lcquad_sft_model(self):
        # loading the tokenizer
        tokenizer = self.load_tokenizer()

        dataset_file_path = self.config['data']["sft_test_dataset"]
        lcquad_data_loader_obj = LCQUADDataHelper(self.config, self.logger)
        test_dataloader = lcquad_data_loader_obj.load_sft_dataloader(tokenizer, dataset_file_path, "test")
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

                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=allowed_max_length,
                    do_sample=False,
                    num_beams=k, # how many outputs are generated
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

                gen_tokens = outputs[:, input_ids.size(1):]
                gen_texts = tokenizer.batch_decode(
                    gen_tokens,
                    skip_special_tokens=True
                )

                generated_rows.extend(
                    {
                        "entity": e,
                        "question": q,
                        "original_sparql": s,
                        "generated_sparql": g.strip(),
                    }
                    for e, q, s, g in zip(
                        batch["entity"],
                        batch["question"],
                        batch["sparql"],
                        gen_texts
                    )
                )

                if batch_idx%500 == 0:
                    self.logger.info(f"output generation is done for batch_idx: {batch_idx}")

        generated_df = pd.DataFrame(generated_rows)

        return generated_df

    def predict_top_K_lcquad_sft_model_helper(self, padding_ind):
        # loading the tokenizer
        tokenizer = self.load_tokenizer()
        tokenizer.padding_side = "left" # during inference default padding is left

        lcquad_model = self.load_lcquad_sft_model()
        lcquad_model.config.use_cache = True # enable KV caching during inference
        lcquad_model.eval()

        lcquad_data_loader_obj = LCQUADDataHelper(self.config, self.logger)

        train_dataset_file_path = self.config['data']["sft_train_dataset"]
        train_dataloader = lcquad_data_loader_obj.load_sft_dataloader(tokenizer, train_dataset_file_path, "train", padding_ind)
        self.logger.info(f"train dataloader {len(train_dataloader)}")

        train_generated_df = self.predict_top_K_lcquad_sft_model(train_dataloader, tokenizer, lcquad_model)
        rm_train_data_path = self.config['data']["rm_train_data"]
        train_generated_df.to_csv(rm_train_data_path, index=False)
        self.logger.info(f"RM output(train) {rm_train_data_path}")


        test_dataset_file_path = self.config['data']["sft_test_dataset"]
        test_dataloader = lcquad_data_loader_obj.load_sft_dataloader(tokenizer, test_dataset_file_path, "test", padding_ind)
        self.logger.info(f"test dataloader {len(test_dataloader)}")

        test_generated_df = self.predict_top_K_lcquad_sft_model(test_dataloader, tokenizer, lcquad_model)
        rm_test_data_path = self.config['data']["rm_test_data"]
        test_generated_df.to_csv(rm_test_data_path, index=False)
        self.logger.info(f"RM output(test) {rm_test_data_path}")


