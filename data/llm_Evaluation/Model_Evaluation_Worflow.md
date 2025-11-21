# LLM Evaluation Workflow

## Generate jsonl's for dataset

In order evlaluate the different LLMs for the model 0 outputs, the files of asciinema output need to be split into chunks so that we can fine tune the model to evaluate streamed data rather than the whole file. This can be split using:

```
cd prepare_data # Located in data/prepare_data
python3 prepare_data.py --inputs ../model_0/inputs --outputs ../model_0/outputs --system_prompt ../../models/model_0/system_prompt.txt --out_dir ../model0_data --tokenizer_model <your_hf_llm_model>
```

## Create a train-test split of the jsonl data

Next, we need to split the model's data into a train and test set that is repeatable using a seed and upload the dataset to huggingface such that it can be loaded by lm_eval (You must be logged into hf auth login). You can upload using:

```
cd llm_Evaluation # Located in data/llm_Evaluation
python3 generate_train_test_set.py --user <hf_username> --dir ../model0_data --dname <educational-ai-agent-small> 
```

Finally, you can move the YAML file data/llm_Evaluation/educationalAITask-Model0.yaml to the directory inside your lm_eval package folder. (move to <your_package_location>/lm_eval/tasks/<new_task_folder>/), and then run the following to benchmark it. In addition to this, you will need to go to the file in <your_package_location>/lm_eval/models/huggingface.py and find the apply_chat_template method. There, please remove the add_generation_prompt part in order to not enforce thinking on the fine tune model (THIS WILL MESS IT UP). With fine tuning regular deepseek, on this prompt it will likely add <think> tokens anyways. Below is the command to run the fine tuned version.

```
lm_eval --model hf --model_args pretrained=<your_hf_base_llm_model>,peft=<fine_tuned_model> --tasks educational-ai-agent-model-0 --device cuda:0 --apply_chat_template --batch_size auto
```

To do so for the prompt testing version for scoring prompts, please move the files metrics.py and llm_judge_metric.py to <your_package_location>/lm_eval/api/ folder. This way, the custom LLM judge metrics will be included. You can then call the prompt testing yaml with:

```
lm_eval --model hf --model_args pretrained=<your_hf_base_llm_model> --tasks educational-ai-agent-model-0-prompt-test --device cuda:0 --apply_chat_template --batch_size auto
```

This should give you the accuracy and the score. To modify the prompt, feel free to do so in the yaml file to not have to do all the reupload steps for the whole dataset.

You are also able to test the model 1 prompting, and model metrics using lm eval. There is an implemented LLM Judge for both the reasoning and the annotation to score them. The rubrics of these are available in the llm_judge_metrics.py file. Move any required yamls from this repository to the location of the lm_eval package. This can be run using the following command:

```
lm_eval --model hf --model_args pretrained=<your_hf_base_llm_model> --tasks educational-ai-agent-model1 --device cuda:0 --apply_chat_template --batch_size auto
```

# NOTE
If the metrics are ever updated in lm_eval that this needs to be updated, only the metric functions for model0 and model1 need to be moved.

CITATION FOR LM EVAL BELOW:

@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {The Language Model Evaluation Harness},
  month        = 07,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.4.3},
  doi          = {10.5281/zenodo.12608602},
  url          = {https://zenodo.org/records/12608602}
}