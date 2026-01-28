from lcquad_finetuning.model.rm.lcquad_rm_model import LCQUADRMModel
from lcquad_finetuning.model.vm.lcquad_vm_model import LCQUADVMModel
from lcquad_finetuning.util.lcquad_exception import LCQUADException
from lcquad_finetuning.util.util_lib import *

class LCQUADMODELHelper:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def load_base_model(self):

        model_path = self.config['model']['base_model_path']
        self.logger.info(f"loading model from {model_path}")
        if self.config['model']['chosen_model'] == "gpt2":
            model_obj = GPT2LMHeadModel.from_pretrained(model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            dtype = self.config['model']['dtype']
            model_obj = AutoModelForCausalLM.from_pretrained(
                                                            model_path,
                                                            dtype=dtype,
                                                            device_map=None)
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        return model_obj

    def load_lcquad_clm_model(self):
        model_path = self.config['model']['clm_model']['clm_model_path']
        self.logger.info(f"loading model from {model_path}")

        if self.config['model']['chosen_model'] == "gpt2":
            model_obj = GPT2LMHeadModel.from_pretrained(model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            dtype = self.config['model']['dtype']
            model_obj = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=dtype,
                device_map=None
            )
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        return model_obj

    def load_lcquad_clm_for_sft_model(self):

        if self.config['model']['chosen_model'] == "gpt2":
            model_path = self.config['model']['clm_model']['clm_model_path']
            self.logger.info(f"loading model from {model_path}")
            model_obj = GPT2LMHeadModel.from_pretrained(model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            # for QLoRA supervised finetuning
            """
            Apple MPS does not support 4-bit / 8-bit quantization
            So we are doing LoRA + fp16 for mac
            """
            model_obj = self.load_lcquad_clm_model()

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
            # model_obj.print_trainable_parameters()
            model_obj.enable_input_require_grads()
            # Enable gradient checkpointing
            model_obj.config.use_cache = False  # Required for checkpointing, On CPU or small GPUs
            model_obj.gradient_checkpointing_enable() # save memory, only if OOM ( Out of Memory )
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        device = self.config['model']['device']
        self.logger.info(f"device:- {device}")
        model_obj.to(device)

        return model_obj

    def load_lcquad_sft_model(self):

        if self.config['model']['chosen_model'] == "gpt2":
            sft_model_path = self.config['model']['sft_model_path']
            model_obj = GPT2LMHeadModel.from_pretrained(sft_model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            clm_model_obj = self.load_lcquad_clm_model()
            sft_model_path = self.config['model']['sft_model']['sft_model_path']
            self.logger.info(f"SFT model loaded from {sft_model_path}")
            model_obj = PeftModel.from_pretrained(
                clm_model_obj,
                sft_model_path
            )
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        device = self.config['model']['device']
        self.logger.info(f"device:- {device}")
        model_obj.to(device)

        return model_obj

    def load_lcquad_clm_for_rm_model(self):
        model_obj = self.load_lcquad_clm_model()

        device = self.config['model']['device']
        self.logger.info(f"device:- {device}")
        model_obj.to(device)

        return model_obj

    def load_reward_model(self):

        rm_model_path = self.config['model']['rm_model']['rm_model_path']
        self.logger.info(f"REWARD model loaded from {rm_model_path}")

        base_model = AutoModelForCausalLM.from_pretrained(rm_model_path)

        rm_model = LCQUADRMModel(base_model, self.config, self.logger)
        device = self.config['model']['device']
        rm_model.head.load_state_dict(
            torch.load(f"{rm_model_path}/reward_head.pt", map_location=device)
        )
        rm_model.to(device)

        return rm_model

    def load_policy_model(self):

        if self.config['model']['chosen_model'] == "gpt2":
            model_path = self.config['model']['sft_model']['sft_model_path']
            self.logger.info(f"loading model from {model_path}")
            model_obj = GPT2LMHeadModel.from_pretrained(model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":

            clm_model_obj = self.load_lcquad_clm_model()
            sft_model_path = self.config['model']['sft_model']['sft_model_path']
            self.logger.info(f"SFT model loaded from {sft_model_path}")
            model_obj = PeftModel.from_pretrained(
                clm_model_obj,
                sft_model_path,
                is_trainable=True # load LORA as trainable
            )
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        model_obj.eval()  # evaluation mode
        device = self.config['model']['device']
        model_obj.to(device)

        return model_obj

    def load_reference_model(self):
        self.logger.info(f"REFERENCE model loaded START")

        clm_model_obj = self.load_lcquad_clm_model()
        sft_model_path = self.config['model']['sft_model']['sft_model_path']
        self.logger.info(f"REFERENCE model loaded from {sft_model_path}")
        ref_model = PeftModel.from_pretrained(
            clm_model_obj,
            sft_model_path
        )
        ref_model.eval() # evaluation mode
        for p in ref_model.parameters():
            p.requires_grad = False
        self.logger.info(f"REFERENCE model loaded END")
        return ref_model

    def load_value_model(self):
        self.logger.info(f"VALUE model loaded START")

        base_model_obj = self.load_lcquad_clm_model()
        value_model_obj = LCQUADVMModel(base_model_obj, self.config, self.logger)
        device = self.config['model']['device']
        self.logger.info(f"device:- {device}")
        value_model_obj.to(device)

        self.logger.info(f"VALUE model loaded END")
        return value_model_obj

    def load_lcquad_inf_model(self):

        if self.config['model']['chosen_model'] == "gpt2":
            inf_model_path = self.config['model']['inf_model']['inf_model_path']
            model_obj = GPT2LMHeadModel.from_pretrained(inf_model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":

            clm_model_obj = self.load_lcquad_clm_model()
            inf_model_path = self.config['model']['inf_model']['inf_model_path']
            self.logger.info(f"Inference model loaded from {inf_model_path}")
            model_obj = PeftModel.from_pretrained(
                clm_model_obj,
                inf_model_path
            )

            # Merge adapter weights for faster inference
            model_obj = model_obj.merge_and_unload()
        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        device = self.config['model']['device']
        self.logger.info(f"device:- {device}")
        model_obj.to(device)

        return model_obj

    def load_model(self, model_ind):

        if model_ind == 'lcquad_base_model':
            return self.load_base_model()
        if model_ind == 'lcquad_clm_model':
            return self.load_lcquad_clm_model()
        elif model_ind == 'lcquad_clm_for_sft_model':
            return self.load_lcquad_clm_for_sft_model()
        elif model_ind == 'lcquad_sft_model':
            return self.load_lcquad_sft_model()
        elif model_ind == 'lcquad_clm_for_rm_model':
            return self.load_lcquad_clm_for_rm_model()
        elif model_ind == 'lcquad_policy_model':
            return self.load_policy_model()
        elif model_ind == 'lcquad_reference_model':
            return self.load_reference_model()
        elif model_ind == 'lcquad_reward_model':
            return self.load_reward_model()
        elif model_ind == 'lcquad_value_model':
            return self.load_value_model()
        elif model_ind == 'lcquad_model_inf':
            return self.load_lcquad_inf_model()
        else:
            msg = f"chosen model_ind is not correct: {model_ind}"
            raise LCQUADException(None, msg)

