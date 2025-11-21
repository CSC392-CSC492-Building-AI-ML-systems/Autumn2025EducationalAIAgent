from openai import OpenAI
import json
import re
import numpy as np
from dotenv import load_dotenv
import os

model1_jprompt_reason = """You are a helpful AI response judge to evaluate model output for annotations of High Level Commands (HLC). You will be given the reasoning output from the model, and the reference reasoning we expect. 

You will score the response on a scale from 1 to 5, with 1 being the worst and 5 being the best, compared to the scores on the given rubric. Your job is to analyze the reasoning and ensure that it outputs logically similar reasoning to the reference reasoning.

Please evaluate how this model's reasoning performs according to the following criteria on the rubric below:

RUBRIC:
Reasoning
- 1: No reasoning or Nonsensical output (E.g. The reasoning is unrelated to reference reasoning, or contains nonexistent words in incoherent ordering). 
- 2: Mostly incorrect explanation of the derivation of the annotation, multiple mistakes .
- 3: Somewhat correct but either over generalizes, has a major mistake, or many small ones that affect the annotation compared to the reference.
- 4: Mostly correct, misses one minor step or slightly overgeneralizes in reasoning which leads to slight error in annotation compared to the reference.
- 5: Clear, logically correct justification that references specific events that ensure a good annotation compared to the reference.

Please output your response with the following format, where x is a score from 1-5:
```
Score: x
```

Model Reasoning:
{reasoning}

Reference Reasoning:
{ref_reasoning}
"""

model1_jprompt_annotation = """You are a helpful AI response judge to evaluate model output for annotations of High Level Commands (HLC). First, you will be given the model's outputted annotation. You will then be given the ground truth reference annotation. You will score the response on a scale from 1 to 5, with 1 being the worst and 5 being the best, compared to the ground truth with scores on the given rubric.

Please evaluate how this model performs according to the following criteria on the rubric below:

RUBRIC:
Dimension 1: Faithfullness to the event (1-5)
- 1: Completely unrelated/hallucinated (Incorrect events described compared to ground truth, etc).
- 2: Vague or partly wrong. User would get a destorted understanding of what is happening (Multiple wrong commands described in annotation compared to ground truth, etc.).
- 3: Gets the high-level idea but omits multiple important actions or includes at most one clear mistake (E.g. Missing the main command in the ground truth annotation).
- 4: Mostly correct, misses minor steps or overgeneralizes slightly (E.g. Only minor commands missing).
- 5: Covers all key actions and transitions; no incosistencies and no major steps missing.

Dimension 2: Granularity and userfulness
- 1: Nonsensical output. The model's annotation is not at all readable (E.g. Not human readable). 
- 2: Outputs an annotation that is largely unhelpful (E.g. "The user ran several commands")
- 3: The annotation is too vague or noisy. Tries to convey a narrative that is unhelpful. (E.g. Narrative is cohesive, but does not capture main details from ground truth, or captures too many minor ones).
- 4: Either too shallow (e.g. "runs some commands") or too low-level (e.g. restates every command that was run)
- 5: Good granularity. Its concise and includes the key intent seen in the ground truth. Describes the events in a similar granularity as ground truth.

Please output your response with the following format, where x,y are scores from 1-5:
```
Dimension 1 Score: x
Dimension 2 Score: y
```

Examples:
Input:
Model Output Annotation:
The user connects via SSH to a remote host

Ground Truth Annotation:
The user copies files to a remote host via SCP
```
Dimension 1 Score: 2
Dimension 2 Score: 3
```
The event that the model described is coherent, but does not relate to the ground truth annotation of SCP. The user would get the wrong impression. so dimension 1 gets a score of 2. 
The granularity for the output annotation language is comparable to ground truth language, but still unrelated. So dimension 2 gets a score of 3.

Input:
Model Output Annotation:
Connect to remote host via SSH and reach the shell prompt.

Ground Truth Annotation:
User initiates SSH connection to server
Output:
```
Dimension 1 Score: 5
Dimension 2 Score: 5
```
The event that the model describes covers all key actions and transitions. There are no incosistencies and no major steps missing. So dimension 1 gets a score of 5.
The model output describes the event without going too in depth or shallow. So dimension 2 gets a score of 5.

Input:
Model Output Annotation:
The user installs the pip packages stored in requirements.txt.

Ground Truth Annotation:
User installs all required packages using pip for the lm_eval repository.
Output:
```
Dimension 1 Score: 4
Dimension 2 Score: 5
```
The event that the model describes misses the repository name which is minor. Otherwise includes major details. So dimension 1 gets a score of 4.
The model output is along the same granularity. So dimension 2 gets a score of 5.

Input:
Model Output Annotation:
Connasdt sdhn saws you to move out

Ground Truth Annotation:
User executes command to run vllm locally for Deepseek R1 Distill.
Output:
```
Dimension 1 Score: 1
Dimension 2 Score: 1
```
The event that the model describes is completely incoherent. So dimension 1 gets a score of 1.
The model output is incoherent. So dimension 2 gets a score of 1.

Input:
Model Output Annotation:
{annotation}

Ground Truth Annotation:
{ref_annotation}
Output:
"""

model0_jprompt_reason = """You are a helpful AI response judge to evaluate model output for groupings of High Level Commands (HLC). You will be given the reasoning output from the model, and the ground truth answer we expect. 

You will score the response on a scale from 1 to 5, with 1 being the worst and 5 being the best, compared to the scores on the given rubric. Your job is to analyze the reasoning and ensure that it outputs logical reasoning to get the correct answer.

Please evaluate how this model's reasoning performs according to the following criteria on the rubric below:

RUBRIC:
Reasoning
- 1: No reasoning or Nonsensical output (E.g. The reasoning is unrelated to reference reasoning, or contains nonexistent words in incoherent ordering). 
- 2: Mostly incorrect explanation of the derivation of the answer (E.g. multiple mistakes, wrong answer, etc).
- 3: Correct outputted answer, but reasoning to get there is mediocre.
- 4: Mostly correct, gives correct answer. Slight error in reasoning, but overall good reasoning.
- 5: Clear, logically correct justification that gets the correct answer.

Please output your response with the following format, where x is a score from 1-5:
```
Score: x
```

Model Reasoning:
{reasoning}

Expected Answer:
{answer}
"""

# Add .env with the url for querying the pod.
# Will be of the form <https://abcdefghijklmn-<port_num>.proxy.runpod.net/v1"
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
pod_url = os.environ.get("POD_URL")

def extract_thinking(pred):
    s = re.findall('<think>(.*)<\\/think>', pred, re.DOTALL)

    if s != []:
        return s[0]
    return None

# Score the reasoning of model 1
def score_reasoning_model0(preds, refs):
    batch = []
    for i in range(len(refs)):
        ref, pred = refs[i], preds[i]
        reasoning = extract_thinking(pred)
        prompt_filled = model0_jprompt_reason.format(reasoning = reasoning, answer = ref)
        batch.append(prompt_filled)
        
    # Get outputs from model
    responses = query_batch(batch) 

    reasoning_scores = []

    for res in responses:
        s = re.findall('Score: ([1-5])', res)
        if s != []:
            reasoning_scores.append(int(s[0]))
        else:
            reasoning_scores.append(0)
    
    return np.array(reasoning_scores)

# Score the reasoning of model 1
def score_reasoning_model1(preds, refs):
    batch = []
    for i in range(len(refs)):
        ref, pred = refs[i], preds[i]
        
        reasoning_pred = ''
        temp = re.findall("(.*)assistantfinal", pred, re.DOTALL)
        if temp != []:
            reasoning_pred = temp[0]
        else:
            temp = re.findall("(.*)\n{", pred, re.DOTALL)
            if temp != []:
                reasoning_pred = temp[0]

        temp = re.findall("(.*)assistantfinal", ref, re.DOTALL)
        reasoning_ref = ''
        if temp != []:
            reasoning_ref = temp[0]
        else:
            temp = re.findall("(.*)\n{", ref, re.DOTALL)
            if temp != []:
                reasoning_ref = temp[0]

        prompt_filled = model1_jprompt_reason.format(reasoning = reasoning_pred, ref_reasoning = reasoning_ref)
        batch.append(prompt_filled)

    # Get outputs from model
    responses = query_batch(batch) 

    reasoning_scores = []

    for res in responses:
        s = re.findall('Score: ([1-5])', res)
        if s != []:
            reasoning_scores.append(int(s[0]))
        else:
            reasoning_scores.append(0)
    
    return np.array(reasoning_scores)

# Score the annotations outputted by model 1
def score_annotations_model1(preds, refs):
    batch = []
    for i in range(len(refs)):
        ref, pred = refs[i], preds[i]
        
        json_str_1 = '{}'
        temp = re.findall('assistantfinal({.*})', ref)
        if temp != []:
            json_str_1 = temp[0]
        else:
           temp = re.findall('\n({.*})', ref)
           if temp != []:
               json_str_1 = temp[0]

        json_str_2 = '{}'

        temp = re.findall('assistantfinal({.*})', pred)
        if temp != []:
            json_str_2 = temp[0]
        else:
           temp = re.findall('\n({.*})', pred)
           if temp != []:
               json_str_2 = temp[0]

        json_1 = json.loads(json_str_1)
        json_2 = json.loads(json_str_2)

        ref_annotation = json_1['annotation']
        pred_annotation = json_2['annotation']

        prompt_filled = model1_jprompt_annotation.format(annotation = pred_annotation, ref_annotation = ref_annotation)

        batch.append(prompt_filled)

    # Get outputs from model
    responses = query_batch(batch)

    event_score = []
    granularity_score = []

    for res in responses:
        s = re.findall('Dimension 1 Score: ([1-5])', res)
        if s != []:
            event_score.append(int(s[0]))
        else:
            event_score.append(0)
        
        s = re.findall('Dimension 2 Score: ([1-5])', res)
        if s != []:
            granularity_score.append(int(s[0]))
        else:
            granularity_score.append(0)
    
    return np.array(event_score), np.array(granularity_score)

# Query judge model Mistral 7B Instruct v0.3 to act as judge LLM model. 
def query_batch(batch: list[str]):
    client = OpenAI(
            api_key="none",
            base_url=pod_url
    )

    res = client.completions.create(
       model="mistralai/Mistral-7B-Instruct-v0.3",
       prompt=batch,
       max_tokens=8192
    )

    return [r.text for r in res.choices]