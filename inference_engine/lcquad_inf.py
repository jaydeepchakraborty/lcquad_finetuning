from lcquad_finetuning.data.lcquad_datahelper import LCQUADDataHelper
from lcquad_finetuning.inference_engine.lcquad_calc_score import LCQuadCalcScore
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer
from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.model.lcquad_modelhelper import LCQUADMODELHelper

class LCQUADInfHelper:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def load_tokenizer(self):
        lcquad_tokenizer_obj = LCQUADTokenizer(self.config, self.logger)
        tokenizer = lcquad_tokenizer_obj.load_tokenizer()
        return tokenizer

    def load_lcquad_inf_model(self):
        lcquad_modelhelper = LCQUADMODELHelper(self.config, self.logger)
        model_obj = lcquad_modelhelper.load_model("lcquad_model_inf")
        return model_obj

    def predict_top_K_lcquad_inf_model(self, dataloader, tokenizer, model, k=1):

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
                        "prompt_without_response": e,
                        "question": q,
                        "original_sparql": s,
                        "generated_sparql": g.strip(),
                    }
                    for e, q, s, g in zip(
                        batch["prompt_without_response"],
                        batch["question"],
                        batch["original_sparql"],
                        gen_texts
                    )
                )

                if batch_idx%500 == 0:
                    self.logger.info(f"output generation is done for batch_idx: {batch_idx}")

        generated_df = pd.DataFrame(generated_rows)

        return generated_df

    def calculate_score(self, lcquad_result_df):
        lcquad_calc_score = LCQuadCalcScore(self.config, self.logger)
        lcquad_calc_score.lcquad_gen_scores(lcquad_result_df)
        return

    def lcquad_test(self):

        padding_ind = "left"

        # loading the tokenizer
        tokenizer = self.load_tokenizer()
        tokenizer.padding_side = padding_ind # during inference default padding is left

        # load inference model
        lcquad_model = self.load_lcquad_inf_model()
        lcquad_model.config.use_cache = True # enable KV caching during inference
        lcquad_model.eval()


        # generating test output
        lcquad_data_loader_obj = LCQUADDataHelper(self.config, self.logger)
        test_dataset_file_path = self.config['data']["inf_test_dataset"]
        test_dataloader = lcquad_data_loader_obj.load_lcquad_inf_dataloader(tokenizer, test_dataset_file_path,
                                                                     "test", padding_ind)
        self.logger.info(f"test dataloader {len(test_dataloader)}")
        test_generated_df = self.predict_top_K_lcquad_inf_model(test_dataloader, tokenizer, lcquad_model)
        inf_test_result_datapath = self.config['data']["inf_result_data"]
        test_generated_df.to_csv(inf_test_result_datapath, index=False)
        self.logger.info(f"Inference output(test) {inf_test_result_datapath}")

        # generate scores for each test sample
        self.calculate_score(test_generated_df)

        return