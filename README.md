# Online Semantic Parsing
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/paper-acl.2022.110-blue.svg)](https://aclanthology.org/2022.acl-long.110)

Online semantic parsing for latency reduction in task-oriented dialogue.

<p align="middle">
  <img width="70%" src="https://user-images.githubusercontent.com/31744226/173400669-bb5f46d6-c12f-4285-b485-f7392edb0f5f.png" />
</p>

<!-- ![pool-party-api-calls](https://user-images.githubusercontent.com/31744226/173400669-bb5f46d6-c12f-4285-b485-f7392edb0f5f.png) -->


Set up the online semantic parsing (OSP) task, formulate the Final Latency Reduction (FLR) evaluation metric, propose the graph-based program prediction model, and develop general frameworks to parse and execute utterances in an online fashion, exploring the balance between latency reduction and program accuracy, as described in the paper
[**Online Semantic Parsing for Latency Reduction in Task-Oriented Dialogue** (ACL 2022)](https://aclanthology.org/2022.acl-long.110/).
If you use any source code or data included in this repository in your work, please cite the following paper:
```bib
@inproceedings{zhou-etal-2022-online,
    title = "Online Semantic Parsing for Latency Reduction in Task-Oriented Dialogue",
    author = "Zhou, Jiawei  and
      Eisner, Jason  and
      Newman, Michael  and
      Platanios, Emmanouil Antonios  and
      Thomson, Sam",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.110",
    doi = "10.18653/v1/2022.acl-long.110",
    pages = "1554--1576",
    abstract = "Standard conversational semantic parsing maps a complete user utterance into an executable program, after which the program is executed to respond to the user. This could be slow when the program contains expensive function calls. We investigate the opportunity to reduce latency by predicting and executing function calls while the user is still speaking. We introduce the task of online semantic parsing for this purpose, with a formal latency reduction metric inspired by simultaneous machine translation. We propose a general framework with first a learned prefix-to-program prediction module, and then a simple yet effective thresholding heuristic for subprogram selection for early execution. Experiments on the SMCalFlow and TreeDST datasets show our approach achieves large latency reduction with good parsing quality, with a 30{\%}{--}65{\%} latency reduction depending on function execution time and allowed cost.",
}
```

<!-- ## Understand Executable Conversational Programs/Graphs

Please read [this document](https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis/blob/master/README-LISPRESS.md) to understand the
syntax of SMCalFlow programs, and read [this document](https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis/blob/master/README-SEMANTICS.md)
to understand their semantics. -->


## Getting Started

### Install
Assuming Conda is the default environment management tool, run
```bash
bash install_with_conda.sh
```

This will set up a local conda environment in the current directory named `./cenv`, with all necessary packages installed.
Activate the environment with `conda activate ./cenv` or `source activate ./cenv` when you need to run the code interactively.

### Code Structure

- `configs*/`: configuration files for data processing, model architectures, and training setups
- `run/`: scripts to build offline full-to-graph models for program/graph generation
- `run_prefix/`: scripts to run experiments for prefix-to-graph model for online semantic parsing
- `run_utter_dlm/`: scripts to run experiments for LM-complete pipeline: de-noising language model + full-to-graph
- `src/`: core code for data structures, parsing state machine, and the action-pointer Transformer model architectures

The core program/graph generation models are developed/extended from the [Action-Pointer Tranformer (APT)](https://aclanthology.org/2021.naacl-main.443.pdf) of a [transition-based parser](https://github.com/IBM/transition-amr-parser), and our PyTorch implementation is formulated as an [extension module](src/fairseq_gap) to the [Fairseq](https://github.com/facebookresearch/fairseq) library.

### Data Download

The default root directories:
- for all raw/processed data -> `../DATA` (accessed through `$DATADIR`)
- for all saved models/results -> `../SAVE` (accessed through `$SAVEDIR`).

These are set (or can be changed) in `set_default_dirs.sh`.

Download the SMCalFlow2.0 data and TreeDST data from [here](https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis/tree/master/datasets), and decompress them in `$DATADIR/smcalflow2.0/` and `$DATADIR/treedst/`, respectively.

For quick testing of a small sample dataset, run `bash scripts/create_sample_dataset.sh`.
This will create a sample dataset of 100 training examples and 50 validation examples from SMCalFlow2.0 data.


## Running Models & Evaluation

All of the commands take the following format
```bash
# run interactively, output logs in the console
bash run*/xxx.sh configs*/config_*/config_xxx.sh [seed, default=42]
```
or
```bash
# send the whole process to the background
bash run*/jbash_xxx.sh configs*/config_*/config_xxx.sh [seed] [gpu_id]
```
where all the data/model/optimization/decoding setups are passed through a `config_xxx.sh` configuration file, and all detailed logs are saved in files.


### Offline Parsing: Train a Program/Graph Prediction Model
---

Run the following script
```bash
# GPU device must be specified if you are working on a multi-GPU machine
CUDA_VISIBLE_DEVICES=0 bash run/run_model_gap.sh configs/config_gap_data_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_ep50/config_model_gap_ptr-lay6-h1_lr0.0005-bsz4096x4-wm4k-dp0.2.sh
```
This would include data processing (use `run/run_data.sh` for processing the data alone), training the model, and decoding for evaluation.

If you want to send the whole process to the background and specify the GPU device, run
```bash
# random seed 42, and GPU ID 1 for example
bash run/jbash_run_model_gap.sh configs/config_gap_data_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_ep50/config_model_gap_ptr-lay6-h1_lr0.0005-bsz4096x4-wm4k-dp0.2.sh 42 1
```

The configuration files in `configs/` are default to SMCalFlow2.0, whereas configurations for TreeDST are in `configs_treedst/`.


*Note: If you are working with no Internet connection and running into the following Fairseq error,*
   ```
   requests.exceptions.ProxyError: HTTPSConnectionPool(host='dl.fbaipublicfiles.com', port=443): Max retries exceeded with url: /fairseq/gpt2_bpe/encoder.json
   ```
   *but you have previously put the necessary files in the default cache directory, check the fix [here](https://github.com/facebookresearch/fairseq/pull/4485/commits/b0e5550a09d425a38eb3d0fe9307483cbaa78340).*


#### Trained Checkpoints

We provide trained offline parsing models for SMCalFlow2.0 and TreeDST data with top-down generation order tested with beam 1 decoding

| Dataset | Valid Exact Match | Model |
|:-------:|:-----------------:|:----------:|
| SMCalFlow2.0 | 80.7 | [Download](https://drive.google.com/file/d/12bASLiPVehG4Lr9JEFCUBNXZ26gVDlM7/view?usp=sharing)  |
| TreeDST | 90.8 | [Download](https://drive.google.com/file/d/19LmKE4UKhOCMglCKaLnjDL8cV_zsT2ZS/view?usp=sharing) |


### Online Parsing: Train a Prefix-to-Graph Model
---
Run the following script
```bash
# GPU device must be specified if you are working on a multi-GPU machine
CUDA_VISIBLE_DEVICES=0 bash run_prefix/run_model_gap_prefix.sh configs/config_gap_data_prefix-last-8ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24/config_model_gap_ptr-lay6-h1_lr0.0001-bsz4096x4-wm4k-dp0.2.sh
```

This would create data consisting of utterance prefixes of last 8 percentages of relative lengths (>= 30%) for training, with full programs masking unseen entities as the targets (use `run_prefix/run_data_xxx.sh` for prefix data processing alone), train the model, and evaluate on all prefix lengths using the validation set.

If you want to send the whole process to the background, use `run_prefix/jbash_run_model_gap_prefix.sh` similarly as in training offline models.
For TreeDST dataset, run with `configs_treedst/` configurations.

**Note:** if you are running the online parser directly from scratch without first running the offline parser, you need to process the offline data first, by running the following script with corresponding data configuration file
```bash
# taking the SMCalFlow data for top-down generation as an example
CUDA_VISIBLE_DEVICES=0 bash run/run_data.sh configs/config_data/config_data_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24.sh
```

#### Trained Checkpoints

We provide trained Prefix-to-Graph models from the above configuration below

| Dataset | Online Parsing Model |
|:-------:|:--------------------:|
| SMCalFlow2.0 | [Download](https://drive.google.com/file/d/1XirN_Vo0Amjlp5aQcdTumwRHPw0_etty/view?usp=sharing) |
| TreeDST | [Download](https://drive.google.com/file/d/1hwEbsumgprL8Vd7trX2RhbS_LZDzJiXt/view?usp=sharing) |



### Online Parsing: Utterance LM-Complete Model from BART
---
Run the following script
```bash
# GPU device must be specified if you are working on a multi-GPU machine
CUDA_VISIBLE_DEVICES=0 bash run_utter_dlm/run_model_utter-dlm.sh configs/config_utter-dlm_src-ct1-npwa/config_dlm_bart-large_data_utter_abs-prefix-all_lr0.00003-scheduler-is-bsz2048x4-wm500-dp0.1-cn0.1_ep1.sh
```
For TreeDST dataset, run with `configs_treedst/` configurations.


### Evaluation: Final Latency Reduction (FLR)
---
Take the Prefix-to-Graph model as an example, assume we want the following execution simulations
- constant utterance token speaking time
- time measured by the number of source tokens
- iterate over different constant execution times
- for each execution time, iterate over different subgraph selection thresholds to get the FLR-cost tradeoff curve

Run the following command (assuming the prefix-to-graph model decoding results are in place)
```bash
bash run_prefix/eval_valid-prefix-allt_latency.sh
```

For other utterance speaking and execution timing models, we can run similarly
- `bash run_prefix/eval_valid-prefix-allt_latency_wr-char-linear.sh`: time measured by milliseconds, word speaking model is a character length linear model
- `bash run_prefix/eval_valid-prefix-allt_latency_wr-real-voice.sh`: time measured by milliseconds, word speaking model fit from real voice ASR data

For running FLR evaluation with different models and/or on different datasets, check `run_prefix/jbash_eval_latency_many*.sh`.

For running FLR evaluation with LM-Complete + Full-to-Graph model, run scripts in `run_utter_dlm/` with similar names as above.
