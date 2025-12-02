# DocStream: Educational AI Agent

DocStream is an open-source system that converts raw, noisy terminal activity into structured, reproducible developer documentation. It processes streamed Asciinema logs, segments them into meaningful events, and generates hierarchical annotations that explain what actually happened in the terminal.

The project is part of an open collaboration between the Human Feedback Foundation, the Linux Foundation, and successive CSC392/492 teams at UofT. Led by Julia Longtin (https://github.com/julialongtin) and Arthur Wolf (https://github.com/arthurwolf) of the Human Feedback Foundation. All data is provided courtesy of Julia and Arthur.

---

## Repository Structure

```
.
├── data/                      # Datasets
│   ├── llm_Evaluation/        # LM eval harness inputs, tasks, and metrics
│   ├── model_0/               # Segmentation datasets
│   │   ├── inputs/            
│   │   ├── outputs/           
│   │   ├── prepare_data/      
│   │   └── raw/               
│   └── model_1/               # Annotation datasets
│       ├── fixed_outputs/     
│       ├── inputs/            
│       └── outputs/           
│
└── models/                    # Training + inference scripts
    ├── model_0/        
    │   ├── runpod/ 
    │   └── scripts/           
    └── model_1/
        └── scripts/
            └── utils/       
```

> **Each model folder contains its own README** with more details and information on how to run.

---

## DocStream Model Details

### Model 0 — Event Segmentation
<img width="2252" height="931" alt="Model 0 pipeline"
     src="https://github.com/user-attachments/assets/ffe58ed4-5b49-49e3-b622-0b1f167256b0" />

* Consumes streamed terminal logs  
* Detects when the user starts a new action  
* Outputs XML-structured “events” suitable for downstream processing  
* Fine-tuned Deepseek Llama 8B Distilled  
* Achieved ~80% segmentation accuracy on a stratified eval set  

### Model 1 — Hierarchical Annotation
<img width="2250" height="955" alt="Model 1 pipeline"
     src="https://github.com/user-attachments/assets/66ae7522-049c-4edc-912b-2f0700c82a1c" />

* Reads Model 0’s event chunks  
* Generates short summaries + depth levels describing goals, subtasks, and resolutions  
* Designed for real-time, incremental annotation  
* Runs on gpt-oss-20b through vLLM  


---

## Evaluation

The repo includes an extended version of EleutherAI’s LM Evaluation Harness.
It provides:

* accuracy metrics for Model 0
* LLM-as-judge scoring for reasoning
* annotation correctness metrics for Model 1
* a reusable structure for future benchmark additions

All of these files are available in the data/llm_Evaluation folder. There is also a worflow file data/llm_Evaluation/Model_Evaluation_Worflow.md that explains how to set up the metrics and tasks, as well as creating the data files and splitting them into a train/test split.

---

## Demo

A small front-end visualization (in the project’s demo folder, if included in your local repo) shows how segmented events flow into annotated summaries, useful for presentations or onboarding. Intended to be hosted by Runpod.

---

## Future Work

* Switch Model 0 to strict NEW/CURRENT classification
* Trim long system-generated event blocks before annotation using similarity threshold filtering.
* Fine-tune Model 1 on expanded reasoning-rich datasets
* Improve LLM-as-judge rubrics
* Integrate the Dockerizer model from previous iteration
* Investigate auto-merging when new documentation conflicts with previous runs

---

## Ethical / Security Notes

Terminal logs may contain sensitive information (paths, IPs, credentials).
The pipeline itself does **not** scrub or redact anything — any deployment must enforce its own filtering before logs are processed.

All datasets used for training are ethically sourced and produced internally.

## Previous Iteration

You can find the previous iteration of this project here: https://github.com/CSC392-CSC492-Building-AI-ML-systems/educational-AI-agent

