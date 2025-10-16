# LLM Evaluation Workflow

## Generate jsonl's for dataset

In order evlaluate the different LLMs for the model 0 outputs, the files of asciinema output need to be split into chunks so that we can fine tune the model to evaluate streamed data rather than the whole file. This can be split using:

```
cd prepare_data # Located in data/prepare_data
python3 prepare_data.py --inputs ../model_0/inputs --outputs ../model_0/outputs --system_prompt ../system_prompt.txt --out_dir ../model0_data --tokenizer_model <your_hf_llm_model>
```

## Create a train-test split of the jsonl data

Next, we need to split the model's data into a train and test set that is repeatable using a seed and upload the dataset to huggingface such that it can be loaded by lm_eval (You must be logged into hf auth login). You can upload using:

```
cd llm_Evaluation # Located in data/llm_Evaluation
python3 generate_train_test_set.py --user <hf_username> --dir ../model0_data --dname <educational-ai-agent-small> 
```

Finally, you can move the YAML file data/llm_Evaluation/educationalAITask.yaml to the directory inside your lm_eval package folder. (move to <your_package_location>/lm_eval/tasks/<new_task_folder>/), and then run the following to benchmark it:

```
lm_eval --model hf --model_args pretrained=<your_hf_llm_model> --tasks educational-ai-agent-small --device cuda:0
```

And record the results.