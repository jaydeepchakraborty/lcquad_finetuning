from lcquad_finetuning.util.util_lib import *
from lcquad_finetuning.util.lcquad_util import LCQuadUtil
from lcquad_finetuning.GPTModelLoader import GPTModelLoader
from lcquad_finetuning.LCQUADDatasetHelper import LCQUADDatasetHelper
from lcquad_finetuning.LCQUADDataLoaderHelper import LCQUADDataLoaderHelper

class LCQUADModelHelper:
    def __init__(self, conf):
        self.config = conf

    def calc_loss_batch(self, input_batch, target_batch, model, device):
        # input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        # logits = model(input_batch)
        # loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

        outputs = model(input_ids=input_batch, labels=target_batch)
        loss = outputs.loss

        return loss

    def calc_loss_loader(self, dataloader, model, device):
        model.eval()
        with torch.no_grad():
            total_loss = 0
            if len(dataloader) == 0:
                return float("nan")

            num_batches = len(dataloader)

            for batch_data in dataloader:
                input_batch, target_batch = batch_data['ip_modf_encoded_tokens'], batch_data['trgt_modf_encoded_tokens']
                loss = self.calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item()

        model.train()
        return total_loss / num_batches


    def train_lcquad_model(self, model, train_loader, val_loader):

        num_epochs = self.config['model']['num_epochs']
        epoch_eval_freq = self.config['model']['epoch_eval_freq']
        batch_eval_freq = self.config['model']['batch_eval_freq']

        device = self.config['model']['device']
        print(f"device:- {device}")

        # Enable gradient checkpointing
        model.config.use_cache = False  # Required for checkpointing, On CPU or small GPUs
        model.gradient_checkpointing_enable()

        # Use Weight Decay but NOT on LayerNorm or Bias
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if "bias" in name or "LayerNorm.weight" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = torch.optim.AdamW([
            {"params": decay, "weight_decay": 0.01},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=5e-5)

        # Use linear warmup + cosine decay (or linear decay).
        # This dramatically improves stability and prevents catastrophic overfitting.
        num_training_steps = num_epochs * len(train_loader)
        num_warmup_steps = int(0.03 * num_training_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        effective_batch_size = self.config['model']['batch_size']['effective_batch_size']  # what you WANT ~ 32
        real_batch_size = self.config['model']['batch_size']['train_batch_size']  # what fits in RAM ~ 8
        accum_steps = effective_batch_size // real_batch_size

        for epoch in range(num_epochs):

            model.train() # set model to training mode
            for batch_id, batch_data in enumerate(train_loader):
                input_batch, target_batch = batch_data['ip_modf_encoded_tokens'].to(device), batch_data['trgt_modf_encoded_tokens'].to(device)

                optimizer.zero_grad()
                loss = self.calc_loss_batch(input_batch, target_batch, model, device)
                loss = loss / accum_steps  # normalize loss
                loss.backward()

                """
                you cannot increase batch size due to memory limits, gradient accumulation is the correct and standard way 
                to reach an effective batch size large enough for stable Transformer training.
                """
                # Gradient Accumulation
                if (batch_id + 1) % accum_steps == 0:
                    # Clip Gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if epoch % epoch_eval_freq == 0 and batch_id % batch_eval_freq == 0:
                    train_loss = self.calc_loss_loader(train_loader, model, device)
                    val_loss = self.calc_loss_loader(val_loader, model, device)
                    print(f"Epoch:- {epoch+1} Batch ID:- {batch_id:06d} Train loss:- {train_loss:3f} Val loss:- {val_loss:3f}")

        return model


    def training_model(self):

        # loading the tokenizer
        tokenizer = LCQuadUtil.get_tokenizer(self.config)

        # load pre-trained model
        model_loader = GPTModelLoader(self.config)
        model_obj = model_loader.load_gpt_model(tokenizer)

        lcquad_data_loader_obj = LCQUADDataLoaderHelper(tokenizer, self.config)

        dataset_file_path = self.config['data']["train_dataset"]
        train_dataloader_obj = lcquad_data_loader_obj.load_dataloader("train", dataset_file_path)
        print(f"train dataloader batches:- {len(train_dataloader_obj)}")

        for batch_id, batch_data in enumerate(train_dataloader_obj):
            org_txt = batch_data['org_txt']
            ip_encoded_text, ip_encoded_tokens = batch_data['ip_encoded_text'], batch_data['ip_encoded_tokens']
            ip_modf_encoded_text, ip_modf_encoded_tokens = batch_data['ip_modf_encoded_text'], batch_data['ip_modf_encoded_tokens']
            trgt_encoded_text, trgt_encoded_tokens = batch_data['trgt_encoded_text'], batch_data['trgt_encoded_tokens']
            trgt_modf_encoded_text, trgt_modf_encoded_tokens = batch_data['trgt_modf_encoded_text'], batch_data['trgt_modf_encoded_tokens']

            print(f"org_txt\n {org_txt}")
            print(f"ip_encoded_text\n {ip_encoded_text}")
            print(f"ip_encoded_tokens\n {ip_encoded_tokens}")
            print(f"ip_modf_encoded_text\n {ip_modf_encoded_text}")
            print(f"ip_modf_encoded_tokens\n {ip_modf_encoded_tokens}")
            print(f"trgt_encoded_text\n {trgt_encoded_text}")
            print(f"trgt_encoded_tokens\n {trgt_encoded_tokens}")
            print(f"trgt_modf_encoded_text\n {trgt_modf_encoded_text}")
            print(f"trgt_modf_encoded_tokens\n {trgt_modf_encoded_tokens}")
            break

        dataset_file_path = self.config['data']["val_dataset"]
        val_dataloader_obj = lcquad_data_loader_obj.load_dataloader("val", dataset_file_path)
        print(f"val dataloader batches:- {len(val_dataloader_obj)}")

        # training the LCQUAD model
        lcquad_gpt_model = self.train_lcquad_model(model_obj, train_dataloader_obj, val_dataloader_obj)

        model_path = self.config['model']['model_path'].replace("{model_ind}", f"{self.config['model']['chosen_model']}")
        self.save_lcquad_model(lcquad_gpt_model, model_path)

        print(f"model saved to {model_path}")

        return


    def save_lcquad_model(self, lcquad_gpt_model, model_path):
        # torch.save(lcquad_gpt_model.state_dict(), model_path)
        lcquad_gpt_model.save_pretrained(model_path)
        return

    def load_lcquad_model(self, model_path):
        # creating model object instance
        # config = self.config['model']['gpt_config']['basic_config']
        # config.update(self.config['model']['gpt_config']['model_config'])
        # model_obj = GPTModel(config)

        # model_obj.load_state_dict(torch.load(model_path, weights_only=True))

        model_obj = GPT2LMHeadModel.from_pretrained(model_path)
        return model_obj

    def test_lcquad_model(self):

        # loading the tokenizer
        tokenizer = LCQuadUtil.get_tokenizer(self.config)

        lcquad_data_loader_obj = LCQUADDataLoaderHelper(tokenizer, self.config)

        dataset_file_path = self.config['data']["test_dataset"]
        test_dataloader_obj = lcquad_data_loader_obj.load_dataloader("test", dataset_file_path)
        print(f"test dataloader {len(test_dataloader_obj)}")

        model_ind = self.config["model"]["chosen_model"]
        model_path = self.config['model']['inf_model_path'].replace("{model_ind}", f"{model_ind}")
        print(f"loading model from {model_path}")
        lcquad_model = self.load_lcquad_model(model_path)

        device = self.config['model']['device']
        print(f"device:- {device}")
        test_loss = self.calc_loss_loader(test_dataloader_obj, lcquad_model, device)
        print(f"test loss:- {test_loss:3f}")

        return

    def text_to_token(self, text, tokenizer):
        text_ids = tokenizer.encode(text, return_tensors="pt", padding=True)
        return text_ids

    def token_to_text(self, token_ids, tokenizer):
        output_text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
        return output_text

    def generate_text(self, model, input_ids, context_size, tokenizer):

        # the maximum number of tokens the model is allowed to generate during inference.
        max_new_tokens = 200
        input_ids_truncated = input_ids[:, -context_size:]

        attention_mask = (input_ids_truncated != tokenizer.pad_token_id).long()

        print(tokenizer.pad_token_id)
        print(tokenizer.eos_token_id)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids_truncated,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

        return output_ids

    def get_sparql(self, query):

        lcquaddatasethelper_obj = LCQUADDatasetHelper(self.config)

        label_eid_map = lcquaddatasethelper_obj.load_lbl_eid_mapping_id()

        sparql = lcquaddatasethelper_obj.modf_ids_entity_helper(query, label_eid_map)

        return sparql

    def predict_ans(self, context_data, tokenizer, model, device):


        strt_context = LCQuadUtil.format_entry(context_data, "test")

        print("User QUERY:-")
        print(strt_context)
        print("===========================")

        context_size = self.config['model']['gpt_config']['basic_config']['allowed_max_length']
        encoded_tokens = self.text_to_token(strt_context, tokenizer).to(device=device)

        print("encoded_tokens:-")
        print(encoded_tokens)
        print("===========================")

        model.eval()
        model_op_token_ids = self.generate_text(model, encoded_tokens, context_size, tokenizer)
        # token_ids = token_ids.to(device=device).detach().clone()

        print("model_op_token_ids:-")
        print(model_op_token_ids)
        print("===========================")

        model_op_text = self.token_to_text(model_op_token_ids, tokenizer)
        print("model_op_text:- ")
        print(model_op_text)
        print("======================")

        model_op_sparql = self.get_sparql(model_op_text)
        print("model_op_sparql output:- ")
        print(model_op_sparql)
        print("\noriginal sparql:-")
        print(context_data['org_sparql'])
        print("======================")


    def inference_lcquad_model(self, test_text):

        model_ind = self.config["model"]["chosen_model"]
        model_path = self.config['model']['inf_model_path'].replace("{model_ind}", f"{model_ind}")
        lcquad_model = self.load_lcquad_model(model_path)

        # loading tokenizer
        tokenizer = LCQuadUtil.get_tokenizer(self.config)

        device = self.config['model']['device']
        self.predict_ans(test_text, tokenizer, lcquad_model, device)