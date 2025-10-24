from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.util.lcquad_util import LCQuadUtil
from lcquad_finetuning.GPTModel import GPTModel
from lcquad_finetuning.GPTModelLoader import GPTModelLoader
from lcquad_finetuning.LCQUADDatasetHelper import LCQUADDatasetHelper
from lcquad_finetuning.LCQUADDataLoaderHelper import LCQUADDataLoaderHelper

class LCQUADModel:
    def __init__(self, conf):
        self.config = conf

    def calc_loss_batch(self, input_batch, target_batch, model, device):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss

    def calc_loss_loader(self, dataloader, model, device):
        model.eval()
        with torch.no_grad():
            total_loss = 0
            if len(dataloader) == 0:
                return float("nan")

            num_batches = len(dataloader)

            for batch_data in dataloader:
                # inputs_txt, targets_txt = batch_data['inputs_txt'], batch_data['targets_txt']
                input_batch, target_batch = batch_data['inputs_tensor'], batch_data['targets_tensor']
                total_loss += self.calc_loss_batch(input_batch, target_batch, model, device)

        model.train()
        return total_loss / num_batches


    def train_lcquad_model(self, model, train_loader, val_loader):

        num_epochs = self.config['model']['num_epochs']
        eval_freq = self.config['model']['eval_freq']

        device = self.config['model']['device']
        print(f"device:- {device}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)

        for epoch in range(num_epochs):

            model.train() # set model to training mode
            for batch_id, batch_data in enumerate(train_loader):
                # org_txt = batch_data['org_txt']
                # inputs_txt, targets_txt = batch_data['inputs_txt'], batch_data['targets_txt']
                input_batch, target_batch = batch_data['inputs_tensor'].to(device), batch_data['targets_tensor'].to(device)

                optimizer.zero_grad()
                loss = self.calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()
                optimizer.step()

                if batch_id % eval_freq == 0:
                    train_loss = self.calc_loss_loader(train_loader, model, device)
                    val_loss = self.calc_loss_loader(val_loader, model, device)
                    print(f"Epoch:- {epoch+1} Batch ID:- {batch_id:06d} Train loss:- {train_loss:3f} Val loss:- {val_loss:3f}")

        return model


    def training_model(self):

        # load pre-trained model
        model_loader = GPTModelLoader(self.config)
        model_obj = model_loader.load_gpt_model()

        lcquad_data_loader_obj = LCQUADDataLoaderHelper(self.config)

        dataset_file_path = self.config['data']["train_dataset"]
        train_dataloader_obj = lcquad_data_loader_obj.load_dataloader("train", dataset_file_path)
        print(f"train dataloader batches:- {len(train_dataloader_obj)}")

        dataset_file_path = self.config['data']["val_dataset"]
        val_dataloader_obj = lcquad_data_loader_obj.load_dataloader("val", dataset_file_path)
        print(f"val dataloader batches:- {len(val_dataloader_obj)}")

        # training the LCQUAD model
        lcquad_gpt_model = self.train_lcquad_model(model_obj, train_dataloader_obj, val_dataloader_obj)

        model_path = self.config['model']['model_path'].replace("{model_ind}", f"{self.config['model']['chosen_model']}")
        torch.save(lcquad_gpt_model.state_dict(), model_path)
        print(f"model saved to {model_path}")

    def load_lcquad_model(self, model_path):
        # creating model object instance
        config = self.config['model']['gpt_config']['basic_config']
        config.update(self.config['model']['gpt_config']['model_config'])
        model_obj = GPTModel(config)

        model_obj.load_state_dict(torch.load(model_path, weights_only=True))
        return model_obj

    def test_lcquad_model(self):

        lcquad_data_loader_obj = LCQUADDataLoaderHelper(self.config)

        dataset_file_path = self.config['data']["test_dataset"]
        test_dataloader_obj = lcquad_data_loader_obj.load_dataloader("test", dataset_file_path)
        print(f"test dataloader {len(test_dataloader_obj)}")

        model_ind = self.config["model"]["chosen_model"]
        model_path = self.config['model']['inf_model_path'].replace("{model_ind}", f"{model_ind}")
        print(f"loading model from {model_path}")
        instruction_model = self.load_lcquad_model(model_path)

        device = self.config['model']['device']
        print(f"device:- {device}")
        test_loss = self.calc_loss_loader(test_dataloader_obj, instruction_model, device)
        print(f"test loss:- {test_loss:3f}")

    def text_to_token(self, text, tokenizer):
        encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        return encoded_tensor

    def token_to_text(self, token_ids, tokenizer):
        flat_tokens = token_ids.squeeze(0).tolist()
        return tokenizer.decode(flat_tokens)

    def generate_text(self, model, idx, context_size, max_new_tokens=50):

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -context_size:]

            with torch.no_grad():
                logits = model(idx_cond)

            logits = logits[:, -1, :]

            probs = torch.softmax(logits, dim=-1)

            idx_next = torch.argmax(probs, dim=-1, keepdim=True)

            if idx_next.item() == 50256:
                break

            idx = torch.cat([idx_cond, idx_next], dim=1)

        return idx

    def get_sparql(self, query):

        lcquaddatasethelper_obj = LCQUADDatasetHelper(self.config)

        label_eid_map = lcquaddatasethelper_obj.load_lbl_eid_mapping_id()

        sparql = lcquaddatasethelper_obj.modf_ids_entity_helper(query, label_eid_map)

        return sparql

    def predict_ans(self, context_data, tokenizer, model, device):
        model.eval()

        strt_context = LCQuadUtil.format_entry(context_data, "test")

        print("start context:-")
        print(strt_context)
        print("===========================")

        context_size = model.pos_emb.weight.shape[0]
        encoded_text = self.text_to_token(strt_context, tokenizer).to(device=device)

        token_ids = self.generate_text(model, encoded_text, context_size)
        token_ids = token_ids.to(device=device).detach().clone()

        decoded_text = self.token_to_text(token_ids, tokenizer)
        print("model output:- ")
        print(decoded_text)
        print("======================")

        sparql = self.get_sparql(decoded_text)
        print("sparql output:- ")
        print(sparql)
        print("\noriginal sparql:-")
        print(context_data['org_sparql'])
        print("======================")


    def inference_instruction_model(self, test_text):

        model_ind = self.config["model"]["chosen_model"]
        model_path = self.config['model']['inf_model_path'].replace("{model_ind}", f"{model_ind}")
        instruction_model = self.load_lcquad_model(model_path)

        # loading tokenizer
        lcquad_util = LCQuadUtil()
        tokenizer = lcquad_util.get_tokenizer(self.config["model"]["tokenizer"])

        device = self.config['model']['device']
        self.predict_ans(test_text, tokenizer, instruction_model, device)