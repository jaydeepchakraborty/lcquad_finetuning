from lcquad_finetuning.util.util_lib import *

class LCQuadDownloadData:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def download_data(self):

        self.logger.info(f"Loading LCQuad init train data from {self.config['data']['init_train_data']}")
        train_df = pd.read_csv(self.config['data']['init_train_data'])
        train_df = train_df[['NNQT_question', 'paraphrased_question', 'question', 'sparql_wikidata']]
        train_df_base = (
            train_df
            .melt(id_vars="sparql_wikidata", value_vars=['NNQT_question', 'paraphrased_question', 'question'],
                  value_name="question_text")
            [["question_text", "sparql_wikidata"]]
        )
        train_df_base.rename(columns={'question_text': 'question', 'sparql_wikidata': 'sparql'}, inplace=True)
        self.logger.info(f"Loaded LCQuad init train data from {self.config['data']['init_train_data']}")

        self.logger.info(f"Saving LCQuad base train data to {self.config['data']['base_train_data']}")
        train_df_base.to_csv(self.config['data']['base_train_data'], index=False)
        self.logger.info(f"Saved LCQuad base train data to {self.config['data']['base_train_data']}")

        self.logger.info(f"Loading LCQuad init test data from {self.config['data']['init_test_data']}")
        test_df = pd.read_csv(self.config['data']['init_test_data'])
        test_df_base = (
            test_df
            .melt(id_vars="sparql_wikidata", value_vars=['NNQT_question', 'paraphrased_question', 'question'],
                  value_name="question_text")
            [["question_text", "sparql_wikidata"]]
        )
        test_df_base.rename(columns={'question_text': 'question', 'sparql_wikidata': 'sparql'}, inplace=True)
        self.logger.info(f"Loaded LCQuad init test data from {self.config['data']['init_test_data']}")

        self.logger.info(f"Saving LCQuad base test data to {self.config['data']['base_test_data']}")
        test_df_base.to_csv(self.config['data']['base_test_data'], index=False)
        self.logger.info(f"Saved LCQuad base train data to {self.config['data']['base_test_data']}")