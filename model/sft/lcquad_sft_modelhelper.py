from lcquad_finetuning.data.lcquad_datahelper import LCQUADDataHelper
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer
from lcquad_finetuning.model.sft.lcquad_sft_model import LCQUADSFTModel

class LCQUADSFTMODELHelper:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def load_tokenizer(self):
        lcquad_tokenizer_obj = LCQUADTokenizer(self.config, self.logger)
        tokenizer = lcquad_tokenizer_obj.load_tokenizer()
        return tokenizer

    def training_lcquad_sft_model(self, ):

        # loading the tokenizer
        tokenizer = self.load_tokenizer()

        lcquad_data_loader_obj = LCQUADDataHelper(self.config, self.logger)
        dataset_file_path = self.config['data']["train_dataset"]
        train_dataloader = lcquad_data_loader_obj.load_sft_dataloader(tokenizer, dataset_file_path, "train")
        self.logger.info(f"train dataloader batches:- {len(train_dataloader)}")

        dataset_file_path = self.config['data']["val_dataset"]
        val_dataloader = lcquad_data_loader_obj.load_sft_dataloader(tokenizer, dataset_file_path, "val")
        self.logger.info(f"val dataloader batches:- {len(val_dataloader)}")

        # training the LCQUAD model
        lcquad_model_sft_obj = LCQUADSFTModel(self.config, self.logger)
        lcquad_sft_model = lcquad_model_sft_obj.train_lcquad_sft_model(train_dataloader, val_dataloader)

        # saving the LCQUAD SFT model
        lcquad_model_sft_obj.save_lcquad_sft_model(lcquad_sft_model)

        return

    def test_lcquad_model(self, dataset_file_path=""):
        # loading the tokenizer
        tokenizer = self.load_tokenizer()

        lcquad_data_loader_obj = LCQUADDataHelper(self.config)

        if dataset_file_path == "":
            dataset_file_path = self.config['data']["test_dataset"]

        test_dataloader = lcquad_data_loader_obj.load_sft_dataloader(tokenizer, dataset_file_path, "test")
        print(f"test dataloader {len(test_dataloader)}")

        lcquad_model_obj = LCQUADSFTModel(self.config, self.logger)
        lcquad_model = lcquad_model_obj.load_lcquad_sft_model()

        device = self.config['model']['device']
        print(f"device:- {device}")
        test_loss = lcquad_model_obj.calc_loss_loader(test_dataloader, lcquad_model, device)
        print(f"test loss:- {test_loss:3f}")

    # def text_to_token(self, text, tokenizer):
    #     text_ids = tokenizer.encode(text, return_tensors="pt", padding=True)
    #     return text_ids
    #
    # def token_to_text(self, token_ids, tokenizer):
    #     output_text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
    #     return output_text
    #
    # def generate_text(self, model, input_ids, context_size, tokenizer):
    #
    #     # the maximum number of tokens the model is allowed to generate during inference.
    #     max_new_tokens = 200
    #     input_ids_truncated = input_ids[:, -context_size:]
    #
    #     attention_mask = (input_ids_truncated != tokenizer.pad_token_id).long()
    #
    #     print(tokenizer.pad_token_id)
    #     print(tokenizer.eos_token_id)
    #
    #     with torch.no_grad():
    #         output_ids = model.generate(
    #             input_ids=input_ids_truncated,
    #             attention_mask=attention_mask,
    #             max_new_tokens=max_new_tokens,
    #             pad_token_id=tokenizer.pad_token_id,
    #             eos_token_id=tokenizer.eos_token_id,
    #             do_sample=False
    #         )
    #
    #     return output_ids
    #
    # def get_sparql(self, query):
    #
    #     lcquaddatasethelper_obj = LCQUADDatasetHelper(self.config)
    #
    #     label_eid_map = lcquaddatasethelper_obj.load_lbl_eid_mapping_id()
    #
    #     sparql = lcquaddatasethelper_obj.modf_ids_entity_helper(query, label_eid_map)
    #
    #     return sparql
    #
    # def predict_ans(self, context_data, tokenizer, model, device):
    #
    #     strt_context = LCQuadUtil.format_entry(context_data, "test")
    #
    #     print("User QUERY:-")
    #     print(strt_context)
    #     print("===========================")
    #
    #     context_size = self.config['model']['gpt_config']['basic_config']['allowed_max_length']
    #     encoded_tokens = self.text_to_token(strt_context, tokenizer).to(device=device)
    #
    #     print("encoded_tokens:-")
    #     print(encoded_tokens)
    #     print("===========================")
    #
    #     model.eval()
    #     model_op_token_ids = self.generate_text(model, encoded_tokens, context_size, tokenizer)
    #     # token_ids = token_ids.to(device=device).detach().clone()
    #
    #     print("model_op_token_ids:-")
    #     print(model_op_token_ids)
    #     print("===========================")
    #
    #     model_op_text = self.token_to_text(model_op_token_ids, tokenizer)
    #     print("model_op_text:- ")
    #     print(model_op_text)
    #     print("======================")
    #
    #     model_op_sparql = self.get_sparql(model_op_text)
    #     print("model_op_sparql output:- ")
    #     print(model_op_sparql)
    #     print("\noriginal sparql:-")
    #     print(context_data['org_sparql'])
    #     print("======================")
    #
    #
    # def inference_lcquad_model(self, test_text):
    #
    #     lcquad_model_obj = LCQUADModel(self.config)
    #     lcquad_model = lcquad_model_obj.load_lcquad_model()
    #
    #     # loading tokenizer
    #     tokenizer = self.load_tokenizer()
    #
    #     device = self.config['model']['device']
    #     self.predict_ans(test_text, tokenizer, lcquad_model, device)

