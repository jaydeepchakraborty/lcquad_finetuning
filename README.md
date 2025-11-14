# lcquad_finetuning

---
## Overview

## Dataset Details
LC-QuAD consists of:

- 5000 natural language questions
- 5000 corresponding SPARQL queries
- Based on **DBpedia v04.16**

For more information, visit the [LC-QuAD GitHub repository](https://github.com/AskNowQA/LC-QuAD).

## Project Flow
### STEP-1
- loading config file
  - lcquad_finetuning/config/lcquad_config.py
- reading, modifying data and saving as Dataset (torch)
  - lcquad_finetuning/LCQUADDatasetHelper

## Installation

