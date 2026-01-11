from lcquad_finetuning.util.lcquad_exception import LCQUADException
from lcquad_finetuning.util.util_lib import *

class LCQUADCLMMODELHelper:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

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

    def load_lcquad_inf_model(self):

        if self.config['model']['chosen_model'] == "gpt2":
            inf_model_path = self.config['model']['inf_model']['inf_model_path']
            model_obj = GPT2LMHeadModel.from_pretrained(inf_model_path)
        elif self.config['model']['chosen_model'] == "Qwen/Qwen2.5-1.5B":
            clm_model_path = self.config['model']['clm_model']['clm_model_path']
            self.logger.info(f"CLM model loaded from {clm_model_path}")

            device = self.config['model']['device']
            self.logger.info(f"device:- {device}")
            # Use float32 for MPS, float16 for CUDA
            dtype = torch.float32 if device == 'mps' else torch.float16

            inf_model_obj = AutoModelForCausalLM.from_pretrained(clm_model_path,
                                                                 torch_dtype=dtype,
                                                                 device_map=device,
                                                                 low_cpu_mem_usage=True)

            inf_model_path = self.config['model']['inf_model']['inf_model_path']
            self.logger.info(f"SFT model loaded from {inf_model_path}")
            model_obj = PeftModel.from_pretrained(
                inf_model_obj,
                inf_model_path
            )

            # Merge adapter weights for faster inference
            model_obj = model_obj.merge_and_unload()
            model_obj.eval()

        else:
            msg = f"chosen model is not correct: {self.config['model']['chosen_model']}"
            self.logger.info(msg)
            raise LCQUADException(None, msg)

        return model_obj

    def load_model(self, model_ind):

        if model_ind == 'lcquad_model_inf':
            return self.load_lcquad_inf_model()
        else:
            msg = f"chosen model_ind is not correct: {model_ind}"
            raise LCQUADException(None, msg)

