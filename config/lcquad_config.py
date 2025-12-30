from lcquad_finetuning.util.util_lib import *

class LCQuadConfig:

    def load_yaml_with_jinja(self, path):

        # 1. Load the raw file using Jinja
        file_loader = FileSystemLoader(searchpath=".")
        env = Environment(loader=file_loader)
        template = env.get_template(path)

        # Step 2 — First render (variables unresolved, but YAML readable)
        first_pass = template.render()

        # Step 3 — Load YAML → get variables
        data = yaml.safe_load(first_pass)

        # Step 4 — Render AGAIN using YAML variables
        second_pass = template.render(**data)

        # Step 5 — Now load final expanded YAML
        return yaml.safe_load(second_pass)

    """
    Loads a YAML file and renders any Jinja2 templates inside it.
    """
    def load_config(self):
        # data configuration
        data_config = self.load_yaml_with_jinja("config/lcquad_data_config.yaml")
        # model configuration
        model_config = self.load_yaml_with_jinja("config/lcquad_model_config.yaml")

        model_config['model']['device'] = torch.device("mps" if torch.mps.is_available() else "cpu")

        model_ind = model_config['model']['chosen_model']
        model_version = "latest"

        base_model_path = model_config['model']['base_model_path'].replace("{model_ind}",model_ind).replace("{model_version}",model_version)
        model_config['model']['base_model_path'] = base_model_path

        clm_model_path = model_config['model']['clm_model_path'].replace("{model_ind}",model_ind).replace("{model_version}",model_version)
        model_config['model']['clm_model_path'] = clm_model_path

        sft_model_path = model_config['model']['sft_model_path'].replace("{model_ind}",model_ind).replace("{model_version}",model_version)
        model_config['model']['sft_model_path'] = sft_model_path

        rm_model_path = model_config['model']['rm_model_path'].replace("{model_ind}",model_ind).replace("{model_version}",model_version)
        model_config['model']['rm_model_path'] = rm_model_path

        rlhf_model_path = model_config['model']['rlhf_model_path'].replace("{model_ind}",model_ind).replace("{model_version}",model_version)
        model_config['model']['rlhf_model_path'] = rlhf_model_path

        inf_model_path = model_config['model']['inf_model_path'].replace("{model_ind}", model_ind)
        model_config['model']['inf_model_path'] = inf_model_path

        config = {"data": data_config['data'], "model": model_config['model']}

        return config
