from lcquad_finetuning.data.lcquad_rlhf_dataloader import LCQuadRLHFDataLoader
from lcquad_finetuning.data.lcquad_rlhf_dataset import LCQUADRLHFDataset
from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.data.lcquad_datapreprocessing import LCQuadDataProcessing
from lcquad_finetuning.tokenizer.lcquad_tokenizer import LCQUADTokenizer
from lcquad_finetuning.data.lcquad_clm_dataset import LCQUADCLMDataset
from lcquad_finetuning.data.lcquad_sft_dataset import LCQUADSFTDataset
from lcquad_finetuning.data.lcquad_sft_dataloader import LCQuadSFTDataLoader

class LCQUADDataHelper:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger


    ############## PREPROCESSING BASE DATA START ##############
    def preprocess_data(self):
        lcquad_data_preprocessing_obj = LCQuadDataProcessing(self.config, self.logger)
        lcquad_data_preprocessing_obj.process_data()
    ############## PREPROCESSING BASE DATA END ##############

    ############## CLM DATA START ##############
    def generate_clm_dataset(self, data_file_df, tokenizer):
        dataset = LCQUADCLMDataset(data_file_df, tokenizer)
        return dataset

    def save_clm_dataset(self, data_set, dataset_path):
        self.logger.info(f"dataset saved at {dataset_path}")
        torch.save(data_set, dataset_path)
        return

    def populate_clm_dataset(self):
        train_df = pd.read_csv(self.config['data']['modf_train_data'])
        valid_df = pd.read_csv(self.config['data']['modf_valid_data'])
        test_df = pd.read_csv(self.config['data']['modf_test_data'])

        df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

        # populating train clm dataset
        tok_obj = LCQUADTokenizer(self.config, logger=self.logger)
        tokenizer = tok_obj.load_tokenizer()
        data_set = self.generate_clm_dataset(df, tokenizer)

        # saving train clm dataset
        dataset_path = self.config['data']['clm_train_dataset']
        self.save_clm_dataset(data_set, dataset_path)

    def load_clm_dataset(self):
        dataset_file_path = self.config['data']['clm_train_dataset']
        self.logger.info(f"loading dataset from:- {dataset_file_path}")
        with torch.serialization.safe_globals([LCQUADCLMDataset]):
            dataset = torch.load(dataset_file_path, weights_only=False)

        return dataset
    ############## CLM DATA END ##############

    ############## SFT DATA START ##############
    def generate_sft_dataset(self, data_file):
        self.logger.info(f"populated datafile from:- {data_file}")
        dataset = LCQUADSFTDataset(data_file)
        return dataset

    def save_sft_dataset(self, data_set, dataset_path):
        self.logger.info(f"dataset saved at {dataset_path}")
        torch.save(data_set, dataset_path)
        return

    def populate_sft_dataset(self):
        file_path = self.config['data']['modf_train_data']
        train_dataset = self.generate_sft_dataset(file_path)
        dataset_path = self.config['data']['train_dataset']
        self.save_sft_dataset(train_dataset, dataset_path)

        file_path = self.config['data']['modf_valid_data']
        valid_dataset = self.generate_sft_dataset(file_path)
        dataset_path = self.config['data']['val_dataset']
        self.save_sft_dataset(valid_dataset, dataset_path)

        file_path = self.config['data']['modf_test_data']
        test_dataset = self.generate_sft_dataset(file_path)
        dataset_path = self.config['data']['test_dataset']
        self.save_sft_dataset(test_dataset, dataset_path)

    def load_sft_dataset(self, dataset_file_path):
        self.logger.info(f"loading dataset from:- {dataset_file_path}")
        with torch.serialization.safe_globals([LCQUADSFTDataset]):
            dataset = torch.load(dataset_file_path, weights_only=False)
        return dataset

    def load_sft_dataloader(self, tokenizer, dataset_file_path, data_set_ind):
        dataset = self.load_sft_dataset(dataset_file_path)
        lcquad_dataloader_obj = LCQuadSFTDataLoader(self.config, self.logger)
        lcquad_dataloader = lcquad_dataloader_obj.load_sft_dataloader(tokenizer, dataset, data_set_ind)
        return lcquad_dataloader
    ############## SFT DATA END ##############

    ############## RM DATA START ##############

    ############## RM DATA END ################

    ############## RLHF-PPO DATA START ##############
    def load_policy_dataset(self, dataset_file_path):
        self.logger.info(f"loading dataset from:- {dataset_file_path}")
        with torch.serialization.safe_globals([LCQUADRLHFDataset]):
            dataset = torch.load(dataset_file_path, weights_only=False)
        return dataset

    def load_policy_dataloader(self, dataset_file_path):
        dataset = self.load_policy_dataset(dataset_file_path)
        lcquad_dataloader_obj = LCQuadRLHFDataLoader(self.config, self.logger)
        lcquad_dataloader = lcquad_dataloader_obj.load_rlhf_dataloader(dataset)
        return lcquad_dataloader
    ############## RLHF-PPO DATA END ##############

