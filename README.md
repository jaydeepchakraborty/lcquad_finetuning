# lcquad_finetuning

---
## Overview

## Dataset Details
LC-QuAD consists of:

- 5000 natural language questions
- 5000 corresponding SPARQL queries
- Based on **DBpedia v04.16**

For more information, visit the [LC-QuAD GitHub repository](https://github.com/AskNowQA/LC-QuAD).
- /Volumes/Jay_4TB/data/LC_Quad/train.csv
- /Volumes/Jay_4TB/data/LC_Quad/test.csv

## Project Flow

### Config file
- lcquad_finetuning/config/lcquad_config.py
  - all the file paths and configurations are mentioned in the config file

### Util file
- lcquad_finetuning/util/util_lib.py
  - all the packages import statements are mentioned in the util file
- lcquad_finetuning/util/lcquad_util.py
  - it contains common functionalities across the project

### Step-1
- reading, modifying data and saving as Dataset (torch)
  - lcquad_finetuning/LCQUADDatasetHelper

#### Step-1.a
- read "sparql_wikidata" column, extract entities such as wd:Q188920, wdt:P31 etc.
- entities like wd:Q188920, wdt:P31 are very specific to SPARQL
- created two maps
  - eid_label_map: ["wd:Q188920": "wd:ENTITY_Q188920", "wdt:P31": "wdt:ENTITY_P31"]
  - /Volumes/Jay_4TB/data/LC_Quad/sparql_wikidata_eid_label_map.json
  - label_eid_map: ["wd:ENTITY_Q188920": "wd:Q188920", "wdt:ENTITY_P31": "wdt:P31"]
  - /Volumes/Jay_4TB/data/LC_Quad/sparql_wikidata_label_eid_map.json
- ["wd:ENTITY_Q188920", "wdt:ENTITY_P31"] are added as new token.
- /Volumes/Jay_4TB/data/LC_Quad/new_token.json

#### Step-1.b
- consider only two columns "paraphrased_question"(renamed question), and "sparql_wikidata"(renamed sparql)
- modify the "sparql" column data with mapped information, 
"wd:Q188920", wdt:P31" is replaced by "wd:ENTITY_Q188920", "wdt:ENTITY_P31"
- here, after modification, train, test and valid data are generated.
- /Volumes/Jay_4TB/data/LC_Quad/modf_train_data.csv
- /Volumes/Jay_4TB/data/LC_Quad/modf_valid_data.csv
- /Volumes/Jay_4TB/data/LC_Quad/modf_test_data.csv

#### Step-1.c
- save GPT2Tokenizer. tokenizer is modified with new tokens
- pad and special tokens are mentioned in the new saved tokenizer
  - pre-modified tokenizer gpt2 with length 50257
  - post-modified tokenizer gpt2 with length 72934
  - /Volumes/Jay_4TB/model_utils/models/LC_Quad/lcquad_tokenizer
  
#### Step-1.d
- creating LCQUADDataset (torch dataset - train, test, valid)
  - each row contains format_data using the tokenizer.
  - /Volumes/Jay_4TB/data/LC_Quad/train_dataset.pt
  - /Volumes/Jay_4TB/data/LC_Quad/valid_dataset.pt
  - /Volumes/Jay_4TB/data/LC_Quad/test_dataset.pt
``` 
format_data = {
            "org_text": format_entry,
            "encoded_text": encoded_tokens,
            "decoded_text": decoded_tokens,
        }
```
  - format_entry is 
```
Question: What is the total list of records discharged by Jerry Lee Lewis?
<SPARQL> 
select distinct ?obj where { wd:Q202729 wdt:P358 ?obj . ?obj wdt:P31 wd:Q273057 }
```
  - org_text is original text from the DBPEDIA dataset transformed to format_entry
  - encoded_text is encoded text using saved tokenizer
  - decoded text is decoded text from the encoded_text using the same tokenizer

### Step-2
- training LCQUAD model.
  - lcquad_finetuning/LCQUADModelHelper.py

#### Step-2.a
- loading the modified tokenizer
  - /Volumes/Jay_4TB/model_utils/models/LC_Quad/lcquad_tokenizer
- loading "gpt2" model and resized to new length (72934) of tokens from tokenizer.

#### Step-2.b
- loading the dataset (torch) and convert it into dataloader (torch)
  - /Volumes/Jay_4TB/data/LC_Quad/train_dataset.pt
  - /Volumes/Jay_4TB/data/LC_Quad/valid_dataset.pt 
  - customized_collate_fn() method in LCQUADDataLoaderHelper class, has the entire loading logic, finally it has below information,
    - org_txt (original input text ~ (Question, SPARQL) without modifying)
    - ip_encoded_tokens (using gpt2 tokenizer on org_text and converted into token-ids)
    - ip_encoded_text (text representation on ip_encoded_tokens)
    - ip_modf_encoded_tokens (modified tokens with 50256('<|endoftext|>'), 50258('<PAD>'))
    - ip_modf_encoded_text (text representation on ip_modf_encoded_tokens)
    - trgt_encoded_tokens (shifted tokens by one place to right)
    - trgt_encoded_text (text representation on trgt_encoded_tokens)
    - trgt_modf_encoded_tokens (replacing everything before <SPARQL> and <PAD> with -100)
    - trgt_modf_encoded_text (text representation on trgt_modf_encoded_tokens, -100 replaced by "IGNORE")
  - ip_modf_encoded_tokens and trgt_modf_encoded_tokens are input to the MODEL (instruction finetuning), the rest are for debugging purpose.
  - Check lcquad_finetuning/debug/input_batch_sample.txt

- Training the MODEL (gpt2)
    - Enable gradient checkpointing
    ```
    model.config.use_cache = False  # Required for checkpointing, On CPU or small GPUs
    model.gradient_checkpointing_enable()
    ```
    - Use Weight Decay but NOT on LayerNorm or Bias
    ```
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if "bias" in name or "LayerNorm.weight" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    ```
    - Use Weight Decay but NOT on LayerNorm or Bias
    ```
            optimizer = torch.optim.AdamW([
                {"params": decay, "weight_decay": 0.01},
                {"params": no_decay, "weight_decay": 0.0},
            ], lr=5e-5)
    ```
    -  Use linear warmup + cosine decay (or linear decay).
    ```
        num_training_steps = num_epochs * len(train_loader)
        num_warmup_steps = int(0.03 * num_training_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    ```
    - gradient accumulation
    ```
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

        return model
    ```
    - saving the trained model 
      - /Volumes/Jay_4TB/model_utils/models/LC_Quad/lcquad_model_gpt2

#### Step-2.c
- INFERENCE using the trained MODEL
  - test_text:
  ```
  test_text = {
        "question": "Which languages does Odia speak?",
        "org_sparql": "SELECT (COUNT(?sub) AS ?value ) { ?sub wdt:P1412 wd:Q33810 }"
    }
  ```
  - loading the trained model 
    - /Volumes/Jay_4TB/model_utils/models/LC_Quad/lcquad_model_gpt2
  - loading the tokenizer
    - /Volumes/Jay_4TB/model_utils/models/LC_Quad/lcquad_tokenizer
    - converting text into token ids 
  - model prediction 
  ```
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
    ```
    - converting token ids to text
    - converting to final SPARQL
      - label_eid_map: ["wd:ENTITY_Q188920": "wd:Q188920", "wdt:ENTITY_P31": "wdt:P31"]
      - /Volumes/Jay_4TB/data/LC_Quad/sparql_wikidata_label_eid_map.json

#### Step-3
- Sample output is as below 
```
User QUERY:-
Question: Which languages does Odia speak?
<SPARQL>
 
===========================
encoded_tokens:-
tensor([[24361,    25,  9022,  8950,   857, 10529,   544,  2740,    30,   198,
         50257,   198,   220]])
===========================
50258
50256
model_op_token_ids:-
tensor([[24361,    25,  9022,  8950,   857, 10529,   544,  2740,    30,   198,
         50257,   198,   220,  7310, 26801,  1391, 60130, 60689, 60689, 60689,
         61646, 61646, 61646, 26801,  5633,  1782, 50256]])
===========================
model_op_text:- 
Question: Which languages does Odia speak?

  distinctobj { wd:ENTITY_Q9043 wdt:ENTITY_P2936 wdt:ENTITY_P2936 wdt:ENTITY_P2936 wdt:ENTITY_P103 wdt:ENTITY_P103 wdt:ENTITY_P103 obj ? }
======================
model_op_sparql output:- 
Question: Which languages does Odia speak?

  distinctobj { wd:ENTITY_Q9043 wdt:ENTITY_P2936 wdt:ENTITY_P2936 wdt:ENTITY_P2936 wdt:ENTITY_P103 wdt:ENTITY_P103 wdt:ENTITY_P103 obj ? }

original sparql:-
SELECT (COUNT(?sub) AS ?value ) { ?sub wdt:P1412 wd:Q33810 }
```


## Installation
- PENDING

