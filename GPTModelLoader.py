from util.util_lib import *

from lcquad_finetuning.GPTModel import GPTModel

class GPTModelLoader:

    def __init__(self, config):
        self.config = config

    def load_gpt2_params_from_tf_ckpt(self, ckpt_path, settings):
        # Initialize parameters dictionary with empty blocks for each layer
        params = {"blocks": [{} for _ in range(settings["n_layer"])]}

        # Iterate over each variable in the checkpoint
        for name, _ in tf.train.list_variables(ckpt_path):
            # Load the variable and remove singleton dimensions
            variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

            # Process the variable name to extract relevant parts
            variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

            # Identify the target dictionary for the variable
            target_dict = params
            if variable_name_parts[0].startswith("h"):
                layer_number = int(variable_name_parts[0][1:])
                target_dict = params["blocks"][layer_number]

            # Recursively access or create nested dictionaries
            for key in variable_name_parts[1:-1]:
                target_dict = target_dict.setdefault(key, {})

            # Assign the variable array to the last key
            last_key = variable_name_parts[-1]
            target_dict[last_key] = variable_array

        return params

    def download_and_load_gpt2(self, model_size, models_dir):

        model_dir = os.path.join(models_dir, model_size)
        # Load settings and params
        tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
        settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
        params = self.load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

        return settings, params

    def assign(self, left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(torch.tensor(right))

    def load_weights_into_gpt(self, gpt, params):
        gpt.pos_emb.weight = self.assign(gpt.pos_emb.weight, params['wpe'])
        gpt.tok_emb.weight = self.assign(gpt.tok_emb.weight, params['wte'])

        for b in range(len(params["blocks"])):
            q_w, k_w, v_w = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.weight = self.assign(
                gpt.trf_blocks[b].att.W_query.weight, q_w.T)
            gpt.trf_blocks[b].att.W_key.weight = self.assign(
                gpt.trf_blocks[b].att.W_key.weight, k_w.T)
            gpt.trf_blocks[b].att.W_value.weight = self.assign(
                gpt.trf_blocks[b].att.W_value.weight, v_w.T)

            q_b, k_b, v_b = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.bias = self.assign(
                gpt.trf_blocks[b].att.W_query.bias, q_b)
            gpt.trf_blocks[b].att.W_key.bias = self.assign(
                gpt.trf_blocks[b].att.W_key.bias, k_b)
            gpt.trf_blocks[b].att.W_value.bias = self.assign(
                gpt.trf_blocks[b].att.W_value.bias, v_b)

            gpt.trf_blocks[b].att.out_proj.weight = self.assign(
                gpt.trf_blocks[b].att.out_proj.weight,
                params["blocks"][b]["attn"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].att.out_proj.bias = self.assign(
                gpt.trf_blocks[b].att.out_proj.bias,
                params["blocks"][b]["attn"]["c_proj"]["b"])

            gpt.trf_blocks[b].ff.layers[0].weight = self.assign(
                gpt.trf_blocks[b].ff.layers[0].weight,
                params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            gpt.trf_blocks[b].ff.layers[0].bias = self.assign(
                gpt.trf_blocks[b].ff.layers[0].bias,
                params["blocks"][b]["mlp"]["c_fc"]["b"])
            gpt.trf_blocks[b].ff.layers[2].weight = self.assign(
                gpt.trf_blocks[b].ff.layers[2].weight,
                params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].ff.layers[2].bias = self.assign(
                gpt.trf_blocks[b].ff.layers[2].bias,
                params["blocks"][b]["mlp"]["c_proj"]["b"])

            gpt.trf_blocks[b].norm1.scale = self.assign(
                gpt.trf_blocks[b].norm1.scale,
                params["blocks"][b]["ln_1"]["g"])
            gpt.trf_blocks[b].norm1.shift = self.assign(
                gpt.trf_blocks[b].norm1.shift,
                params["blocks"][b]["ln_1"]["b"])
            gpt.trf_blocks[b].norm2.scale = self.assign(
                gpt.trf_blocks[b].norm2.scale,
                params["blocks"][b]["ln_2"]["g"])
            gpt.trf_blocks[b].norm2.shift = self.assign(
                gpt.trf_blocks[b].norm2.shift,
                params["blocks"][b]["ln_2"]["b"])

        gpt.final_norm.scale = self.assign(gpt.final_norm.scale, params["g"])
        gpt.final_norm.shift = self.assign(gpt.final_norm.shift, params["b"])
        gpt.out_head.weight = self.assign(gpt.out_head.weight, params["wte"])

        return gpt

    def load_gpt_model_helper(self, model_config, pre_train_wt_ind):

        # creating model object instance
        config = model_config['basic_config']
        config.update(model_config['model_config'])
        model_obj = GPTModel(config)

        if pre_train_wt_ind:
            # loading pre-trained GPT model
            model_size, model_dir = model_config['model_size'], model_config['models_dir']
            settings, params = self.download_and_load_gpt2(model_size, model_dir)

            model_obj = self.load_weights_into_gpt(model_obj, params)

        return model_obj


    def load_gpt_model(self, pre_train_wt_ind=True):

        # load pre-trained model
        model_config = self.config['model']["gpt_config"]
        model_obj = self.load_gpt_model_helper(model_config, pre_train_wt_ind)
        device = self.config['model']['device']
        model_obj.to(device)

        print(f'pre-trained model ind:- { self.config["model"]["chosen_model"]}')

        return model_obj