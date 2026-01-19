from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.data.lcquad_format_entry import LCQuadFormatEntry

class LCQuadDataProcessing:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def clean_text_column(self, df, column_name):
        # 1. Convert to lowercase for consistency
        df[column_name] = df[column_name].str.lower()

        # 2. Remove leading/trailing whitespace
        df[column_name] = df[column_name].str.strip()

        # 3. Remove punctuation
        # string.punctuation includes !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        lcquad_punctuation = r"""!"#$%&'()*+,-/:;<=>?@[\]^_`{|}~"""
        df[column_name] = df[column_name].str.replace(f'[{lcquad_punctuation}]', '', regex=True)

        # 4. Remove extra whitespace between words
        df[column_name] = df[column_name].str.replace(r'\s+', ' ', regex=True).str.strip()

        return df

    def process_data(self):

        train_df = pd.read_csv(self.config['data']['base_train_data'])
        train_df = self.clean_text_column(train_df, 'question')
        train_df["prompt_with_response"] = train_df.apply(lambda x: LCQuadFormatEntry.prompt_format_entry(x, "prompt_with_response"), axis=1)
        train_df["prompt_without_response"] = train_df.apply(lambda x: LCQuadFormatEntry.prompt_format_entry(x, "prompt_without_response"), axis=1)

        train_df, valid_df = train_test_split(train_df, test_size=0.1, random_state=42)
        train_df.to_csv(self.config['data']['modf_train_data'], index=False)
        self.logger.info(f"modified train data is saved to "
                         f"{self.config['data']['modf_train_data']}, "
                         f"train-shape: {train_df.shape}")

        valid_df.to_csv(self.config['data']['modf_valid_data'], index=False)
        self.logger.info(f"modified valid data is saved to "
                         f"{self.config['data']['modf_valid_data']}, "
                         f"valid-shape: {valid_df.shape}")

        test_df = pd.read_csv(self.config['data']['base_test_data'])
        test_df = self.clean_text_column(test_df, 'question')
        test_df["prompt_with_response"] = test_df.apply(lambda x: LCQuadFormatEntry.prompt_format_entry(x, "prompt_with_response"), axis=1)
        test_df["prompt_without_response"] = test_df.apply(lambda x: LCQuadFormatEntry.prompt_format_entry(x, "prompt_without_response"), axis=1)

        test_df.to_csv(self.config['data']['modf_test_data'], index=False)
        self.logger.info(f"modified test data is saved to "
                         f"{self.config['data']['modf_test_data']}, "
                         f"test-shape: {test_df.shape}")

        return

