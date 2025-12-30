from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.data.lcquad_format_entry import LCQuadFormatEntry

class LCQuadDataProcessing:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def process_data(self):

        train_df = pd.read_csv(self.config['data']['base_train_data'])
        train_df["entry"] = train_df.apply(lambda x: LCQuadFormatEntry.format_entry(x, "train"), axis=1)

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
        test_df["entry"] = test_df.apply(lambda x: LCQuadFormatEntry.format_entry(x, "test"), axis=1)
        test_df.to_csv(self.config['data']['modf_test_data'], index=False)
        self.logger.info(f"modified test data is saved to "
                         f"{self.config['data']['modf_test_data']}, "
                         f"test-shape: {test_df.shape}")

        return

