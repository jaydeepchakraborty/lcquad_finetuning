from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.LCQUADDataset import LCQUADDataset
from lcquad_finetuning.util.lcquad_util import LCQuadUtil

class LCQUADDatasetHelper:

    def __init__(self, config: dict):
        self.config = config

    def populate_labels(self, eid: int, e_ids_len: int):
        padded_zero =  len(str(e_ids_len)) - len(str(eid))
        return "ENTITY_"+padded_zero*"0"+str(eid)

    def save_mapping_id(self):
        train_df = pd.read_csv(self.config['data']['train_data'])
        train_df = train_df[['sparql_wikidata']]
        test_df = pd.read_csv(self.config['data']['test_data'])
        test_df = test_df[['sparql_wikidata']]

        df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

        eid_pattern = re.compile(r'[a-zA-Z]+:[a-zA-Z]+\d+')

        eids = set()
        for sparql in df['sparql_wikidata'].dropna():
            eids.update(eid_pattern.findall(sparql))

        e_ids_len = len(eids)
        eid_label_map = {"ENTITY_UNK": "ENTITY_"+len(str(e_ids_len))*"0"}
        label_eid_map = {"ENTITY_"+len(str(e_ids_len))*"0": "ENTITY_UNK"}
        e_idx = 1

        for eid in tqdm(eids, desc="ENTITY_ID fetching"):
            label = self.populate_labels(e_idx, e_ids_len)
            eid_label_map[eid] = label
            label_eid_map[label] = eid
            e_idx += 1

        with open(self.config['data']['sparql_wikidata_eids_labels_mapping'], "w") as f:
            json.dump(eid_label_map, f, indent=2)

        with open(self.config['data']['sparql_wikidata_labels_eids_mapping'], "w") as f:
            json.dump(label_eid_map, f, indent=2)

        return

    def load_eid_lbl_mapping_id(self):
        eid_label_map = {}
        with open(self.config['data']['sparql_wikidata_eids_labels_mapping'], "r") as f:
            eid_label_map = json.load(f)
        return eid_label_map

    def load_lbl_eid_mapping_id(self):
        label_eid_map = {}
        with open(self.config['data']['sparql_wikidata_labels_eids_mapping'], "r") as f:
            label_eid_map = json.load(f)
        return label_eid_map

    def modf_entity_ids_helper(self, df, eid_lbl_mapping):

        eid_pattern = re.compile(r'[a-zA-Z]+:[a-zA-Z]+\d+')

        def modf_entity_ids(text, replacement_dict):
            matches = set(eid_pattern.findall(text))  # unique IDs in string
            for match in matches:
                if match in replacement_dict:
                    pattern = r'\b' + re.escape(match) + r'\b'
                    text = re.sub(pattern, replacement_dict[match], text)
            return text

        df["sparql_modf"] = df["sparql"].apply(lambda x: modf_entity_ids(x, eid_lbl_mapping))

        return df

    def modf_ids_entity_helper(self, query, lbl_eid_mapping):

        # 1. Extract all entity identifiers from the query
        entities = re.findall(r"\bENTITY_\d+\b", query)

        # 2. Replace each entity with its value (if found in dict)
        for entity in entities:
            if entity in lbl_eid_mapping:
                query = query.replace(entity, lbl_eid_mapping[entity])

        return query


    def modf_lcquad_data(self):

        eid_lbl_mapping = self.load_eid_lbl_mapping_id()

        train_df = pd.read_csv(self.config['data']['train_data'])
        train_df = train_df[['paraphrased_question', 'sparql_wikidata']]
        train_df.rename(columns={'paraphrased_question': 'question', 'sparql_wikidata': 'sparql'}, inplace=True)
        train_df = self.modf_entity_ids_helper(train_df, eid_lbl_mapping)
        print(train_df.shape)
        print(train_df.head())

        train_df, valid_df = train_test_split(train_df, test_size=0.1, random_state=42)
        train_df.to_csv(self.config['data']['modf_train_data'], index=False)
        valid_df.to_csv(self.config['data']['modf_valid_data'], index=False)

        test_df = pd.read_csv(self.config['data']['test_data'])
        test_df = test_df[['paraphrased_question', 'sparql_wikidata']]
        test_df.rename(columns={'paraphrased_question': 'question', 'sparql_wikidata': 'sparql'}, inplace=True)
        test_df = self.modf_entity_ids_helper(test_df, eid_lbl_mapping)
        print(test_df.shape)
        print(test_df.head())
        test_df.to_csv(self.config['data']['modf_test_data'], index=False)

        return

    def populate_dataset(self, json_file, tokenizer):
        dataset = LCQUADDataset(json_file, tokenizer)
        return dataset

    def save_dataset(self, data_set, dataset_path):
        torch.save(data_set, dataset_path)

    def load_dataset(self, dataset_file_path):
        with torch.serialization.safe_globals([LCQUADDataset]):
            dataset = torch.load(dataset_file_path, weights_only=False)

        return dataset

    def prepare_data(self):
        # Step-1: saving mapped ids
        # self.save_mapping_id()

        # Step-2: loading mapped ids
        # self.load_eid_lbl_mapping_id()

        # Step-3: modify data (SPARQL ENTRY)
        # self.modf_lcquad_data()

        # Step-4: populate dataset
        lcquad_util = LCQuadUtil()
        tokenizer = lcquad_util.get_tokenizer(self.config['model']['tokenizer'])  # loading tokenizer

        file_path = self.config['data']['modf_train_data']
        train_dataset = self.populate_dataset(file_path, tokenizer)
        dataset_path = self.config['data']['train_dataset']
        self.save_dataset(train_dataset, dataset_path)

        for data in self.load_dataset(dataset_path):
            print(data)
            break

        file_path = self.config['data']['modf_valid_data']
        valid_dataset = self.populate_dataset(file_path, tokenizer)
        dataset_path = self.config['data']['val_dataset']
        self.save_dataset(valid_dataset, dataset_path)

        file_path = self.config['data']['modf_test_data']
        test_dataset = self.populate_dataset(file_path, tokenizer)
        dataset_path = self.config['data']['test_dataset']
        self.save_dataset(test_dataset, dataset_path)

        return


