#!/usr/bin/env python3
"""
FastAPI server for real-time visualization of Model 0 and Model 1 pipelines.
Streams processing updates via WebSocket.
"""

import asyncio
import os
import re
import sys
import time
from typing import List, Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from datasets import load_dataset
import requests
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

# Model 1 configuration (matching model1test.py)
TAIL_K = 8  # Number of neighbors to send to Model 1

app = FastAPI()

# Global state
current_group_events: List[str] = []
current_group_id: int = 0
model0_instance = None
model1_instance = None
processing_complete = False
processing_in_progress = False

# Store all messages for replay to new clients
message_history: List[Dict] = []

# Store all connected WebSocket clients
connected_clients: List[WebSocket] = []

# Store Model 1 history (matches model1test.py structure)
model1_history: List[Dict] = []  # [{"gid": int, "annotation": str, "depth": int}]
model1_system_prompt: str = ""  # Loaded from file

sortme_pattern = re.compile(
    r"<(?:user_input|system_output)\b[^>]*?sortme=\"True\"[^>]*?(?:/>|>.*?</(?:user_input|system_output)>)",
    flags=re.DOTALL,
)


def compute_curr_depth_from_history(history: List[Dict]) -> int:
    """Compute current depth from neighbor history - matches model1test.py"""
    curr = 0
    for h in history:
        d = h.get("depth")
        if d is None:
            continue
        if d == -1:
            curr -= 1
        elif isinstance(d, int) and d > 0:
            curr += d
    if curr > 0:
        curr = 0
    return curr


def build_model1_inputs_block(
    group_id: int, group_xml_inner: str, history: List[Dict]
) -> str:
    """Build the <inputs> block for Model 1 - matches model1test.py exactly"""
    # Get last TAIL_K neighbors
    neighbors_tail = history[-TAIL_K:] if len(history) > 0 else []

    # Build neighbors XML
    neighbors_xml = (
        "\n".join(
            [
                f'        <neighbor id="{t["gid"]}" depth="{t.get("depth", 0)}">{t.get("annotation", "")}</neighbor>'
                for t in neighbors_tail
            ]
        )
        or "        <neighbor>(none)</neighbor>"
    )

    # Build targets XML
    targets_xml = f'        <target id="{group_id}">\n          <event>\n{group_xml_inner}\n          </event>\n        </target>'

    # Compute current depth
    curr_depth_val = compute_curr_depth_from_history(history)

    # Build the exact <inputs> block
    user_block = f"""<inputs>
  <curr_depth_max>{curr_depth_val}</curr_depth_max>
  <neighbors>
{neighbors_xml}
  </neighbors>
  <target_events>
{targets_xml}
  </target_events>
</inputs>""".strip()

    return user_block


def send_vllm_request(url, headers, payload, max_retries=3):
    """Send request to vLLM worker with retry logic and error handling"""
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=300)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print(f"Request timed out. Attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print("Rate limit exceeded. Waiting before retry...")
                time.sleep(5)
            elif e.response.status_code >= 500:
                print(f"Server error: {e.response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
            else:
                raise
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)

    raise Exception("Max retries exceeded")


class Model1:
    def __init__(self):
        print("Initializing Model 1 RunPod client...")

        # Get RunPod configuration from environment
        self.api_key = os.environ.get("MODEL1_API_KEY")
        self.endpoint_id = os.environ.get("MODEL1_ENDPOINT_ID")

        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY environment variable not set")
        if not self.endpoint_id:
            raise ValueError("MODEL1_ENDPOINT_ID environment variable not set")

        self.run_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/run"
        self.status_url_base = f"https://api.runpod.ai/v2/{self.endpoint_id}/status"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        print(f"Model 1 configured for endpoint: {self.endpoint_id}")

    def submit_job(self, messages: List[Dict[str, str]]) -> str:
        """Submit async job and return job ID"""
        payload = {"input": {"messages": messages}}

        print(f"Submitting Model 1 job with {len(messages)} messages")

        try:
            result = send_vllm_request(
                self.run_url, self.headers, payload, max_retries=3
            )
            job_id = result.get("id")
            if not job_id:
                raise Exception(f"No job ID in response: {result}")
            print(f"Model 1 job submitted: {job_id}")
            return job_id

        except Exception as e:
            error_msg = f"Model 1 job submission failed: {str(e)}"
            print(error_msg)
            raise

    def poll_status(self, job_id: str, max_attempts: int = 60) -> Dict:
        """Poll for job completion, returns job result"""
        print(f"Polling Model 1 job: {job_id}")

        for attempt in range(max_attempts):
            try:
                response = requests.get(
                    f"{self.status_url_base}/{job_id}", headers=self.headers, timeout=10
                )
                response.raise_for_status()
                data = response.json()

                status = data.get("status")

                if status == "COMPLETED":
                    print(f"Model 1 job {job_id} completed")
                    return data.get("output", {})
                elif status == "FAILED":
                    error = data.get("error", "Unknown error")
                    raise Exception(f"Job failed: {error}")
                elif status in ["CANCELLED", "TIMED_OUT"]:
                    raise Exception(f"Job {status.lower()}")

                # Still processing, wait before next poll
                time.sleep(2)

            except requests.exceptions.RequestException as e:
                print(f"Error polling status: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2)
                else:
                    raise

        raise TimeoutError(f"Job polling exceeded {max_attempts} attempts")


class Model0:
    def __init__(self):
        print("Initializing Model 0 vLLM client...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        )

        self.base_url = os.environ.get("MODEL0_BASE_URL")
        self.api_key = os.environ.get("MODEL0_API_KEY")

        self.url = f"{self.base_url}/v1/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        print(f"Model 0 configured for endpoint: {self.base_url}")

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompt = formatted_prompt.replace("<think>", "")

        print(formatted_prompt)

        payload = {
            "model": "educational",
            "prompt": formatted_prompt,
            "temperature": 0.5,
            "max_tokens": 512,
        }

        print(f"Calling Model 0 API with prompt length: {len(prompt)}")

        try:
            result = send_vllm_request(self.url, self.headers, payload)

            if "choices" in result and len(result["choices"]) > 0:
                generated_text = result["choices"][0]["text"]
                generated_text = generated_text.replace("</think>", "").strip()
                print(f"Model 0 response received: {len(generated_text)} chars")
                return generated_text.strip()
            else:
                print(f"Unexpected response format: {result}")
                return "ERROR: Unexpected response format"

        except Exception as e:
            error_msg = f"Model 0 generation failed: {str(e)}"
            print(error_msg)
            return f"ERROR: {error_msg}"
        return "a"


def extract_clean_event(event_with_attrs: str) -> str:
    return re.sub(r'\s+(group|sortme)="[^"]*"', "", event_with_attrs)


def parse_model0_response(text: str) -> str:
    match = re.search(r"Answer:\s*(NEW|\d+)", text, re.IGNORECASE)

    if not match:
        return "UNKNOWN"

    answer = match.group(1)
    if answer.upper() == "NEW":
        return "NEW"
    else:
        return "EXISTING"


async def broadcast_message(message: Dict):
    global message_history, connected_clients

    message_history.append(message)

    disconnected = []
    for client in connected_clients:
        try:
            await client.send_json(message)
        except:
            disconnected.append(client)

    for client in disconnected:
        connected_clients.remove(client)


async def process_model1(group_id: int, events: List[str]):
    """Process a single group with Model 1 in the background - matches model1test.py"""
    global model1_instance, model1_history, model1_system_prompt

    try:
        # Format events as XML inner content
        group_xml_inner = "\n".join(events)

        # Get current depth for display
        curr_depth_val = compute_curr_depth_from_history(model1_history)

        # Format neighbors for frontend display
        neighbors_tail = model1_history[-TAIL_K:] if model1_history else []
        neighbor_display = [
            f"- id={t['gid']} depth={t.get('depth', 0)} annotation={t.get('annotation', '')}"
            for t in neighbors_tail
        ]

        # Send to frontend showing input
        await broadcast_message(
            {
                "type": "model1_input",
                "data": {
                    "group_id": group_id,
                    "neighbors": neighbor_display,
                    "events": events,
                    "current_depth": curr_depth_val,
                },
            }
        )

        # Build the <inputs> block using exact same logic as model1test.py
        user_content = build_model1_inputs_block(
            group_id, group_xml_inner, model1_history
        )

        # Build messages with system prompt from file
        messages = [
            {"role": "system", "content": model1_system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Submit job to RunPod
        job_id = await asyncio.to_thread(model1_instance.submit_job, messages)

        # Poll for result
        result = await asyncio.to_thread(model1_instance.poll_status, job_id)

        # Extract response
        json_output = result.get("json", {})
        thinking = result.get("thinking", "")
        full_text = result.get("text", "")

        # Handle case where JSON parsing failed and returned error dict
        if "error" in json_output and json_output.get("error") == "no_valid_json":
            # Try to extract JSON from raw text
            raw_text = json_output.get("raw", "")
            print(
                f"JSON parsing failed, attempting to extract from raw text: {raw_text[:200]}..."
            )

            import json as json_module

            # First try to find markdown code blocks with JSON using regex
            markdown_match = re.search(
                r"```(?:json)?\s*\n(.*?)\n```", raw_text, re.DOTALL
            )
            if markdown_match:
                print("Found JSON in markdown code block")
                json_text = markdown_match.group(1).strip()
                try:
                    json_output = json_module.loads(json_text)
                    print(f"Successfully extracted JSON from markdown: {json_output}")
                except Exception as e:
                    print(f"Failed to parse markdown JSON: {e}")
                    json_output = {
                        "annotation": "Failed to parse model response",
                        "depth": 0,
                    }
            else:
                # Fallback: Remove markdown code blocks if text starts with them
                cleaned = raw_text.strip()
                if cleaned.startswith("```"):
                    # Remove ```json and trailing ```
                    lines = cleaned.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    cleaned = "\n".join(lines)

                try:
                    json_output = json_module.loads(cleaned.strip())
                    print(f"Successfully extracted JSON: {json_output}")
                except Exception as e:
                    print(f"Failed to extract JSON: {e}")
                    json_output = {
                        "annotation": "Failed to parse model response",
                        "depth": 0,
                    }

        annotation = json_output.get("annotation", "No summary generated")
        depth = json_output.get("depth", 0)

        # Update rolling history (matches model1test.py)
        model1_history.append(
            {"gid": group_id, "annotation": annotation, "depth": depth}
        )

        # Send result to frontend
        await broadcast_message(
            {
                "type": "model1_summary",
                "data": {
                    "group_id": group_id,
                    "summary": annotation,
                    "depth": depth,
                    "thinking": thinking,
                    "json": json_output,
                },
            }
        )

    except Exception as e:
        error_msg = f"Error in Model 1 processing: {str(e)}"
        print(error_msg)
        await broadcast_message(
            {"type": "model1_error", "data": {"group_id": group_id, "error": error_msg}}
        )


async def process_model0_pipeline(dataset, system_prompt: str, model1_sys_prompt: str):
    """Process Model 0 pipeline - continues while Model 1 runs in background"""
    global current_group_events, current_group_id, model0_instance, model1_instance
    global \
        processing_complete, \
        processing_in_progress, \
        model1_history, \
        model1_system_prompt

    if processing_in_progress or processing_complete:
        return

    processing_in_progress = True

    # Use provided Model 1 system prompt
    model1_system_prompt = model1_sys_prompt

    model0_instance = Model0()
    model1_instance = Model1()
    model1_history = []

    current_group_events = []
    current_group_id = 0

    for idx, example in enumerate(dataset["train"]):
        input_xml = example["input"]
        expected_output = example["output"]

        # Extract the new event
        sortme_match = sortme_pattern.search(input_xml)
        if not sortme_match:
            continue

        sortme_event = sortme_match.group(0)
        clean_event = extract_clean_event(sortme_event)

        await broadcast_message({"type": "raw_event", "data": clean_event})

        await broadcast_message(
            {
                "type": "model0_input",
                "data": {
                    "input_xml": input_xml,
                    "current_event": clean_event,
                    "group_id": current_group_id,
                },
            }
        )

        # Generate Model 0 response
        prompt = f"{system_prompt}\n\n{input_xml}"
        response = model0_instance.generate(prompt)

        # Send Model 0 full response for visualization
        await broadcast_message(
            {"type": "model0_thinking", "data": {"full_response": response}}
        )

        prediction_type = parse_model0_response(expected_output)

        # Send Model 0 answer for visualization
        await broadcast_message(
            {
                "type": "model0_answer",
                "data": {"answer": prediction_type, "group_id": current_group_id},
            }
        )

        if prediction_type == "NEW":
            if current_group_events:
                asyncio.create_task(
                    process_model1(current_group_id, current_group_events.copy())
                )

            current_group_id += 1
            current_group_events = [clean_event]
        else:
            current_group_events.append(clean_event)

        await broadcast_message(
            {
                "type": "current_group_update",
                "data": {"group_id": current_group_id, "events": current_group_events},
            }
        )

        await asyncio.sleep(0.3)

    if current_group_events:
        asyncio.create_task(
            process_model1(current_group_id, current_group_events.copy())
        )

    await broadcast_message({"type": "complete"})
    processing_complete = True
    processing_in_progress = False


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming updates"""
    global \
        connected_clients, \
        message_history, \
        processing_complete, \
        processing_in_progress

    await websocket.accept()
    connected_clients.append(websocket)

    try:
        # If we already have history, replay it to the new client
        if message_history:
            print(f"Replaying {len(message_history)} messages to new client")
            for message in message_history:
                await websocket.send_json(message)

        if not processing_complete and not processing_in_progress:
            dataset = load_dataset(
                "patea4/educational-ai-agent-small", data_files="1721655754.jsonl"
            )

            # Load Model 0 system prompt
            model0_prompt_file = os.environ.get(
                "MODEL0_SYSTEM_PROMPT", "model_0/system_prompt.txt"
            )
            model0_prompt_path = Path(__file__).parent / model0_prompt_file
            model0_system_prompt = model0_prompt_path.read_text()

            # Load Model 1 system prompt (can be overridden via env var)
            model1_prompt_file = os.environ.get(
                "MODEL1_SYSTEM_PROMPT", "model_1/model1_system_prompt.txt"
            )
            model1_prompt_path = Path(__file__).parent / model1_prompt_file
            model1_system_prompt = model1_prompt_path.read_text()

            asyncio.create_task(
                process_model0_pipeline(
                    dataset, model0_system_prompt, model1_system_prompt
                )
            )

        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        try:
            await websocket.send_json({"type": "error", "data": str(e)})
        except:
            pass
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


@app.get("/")
async def get():
    """Serve the main HTML page"""
    html_path = Path(__file__).parent / "demo_frontend.html"
    return HTMLResponse(content=html_path.read_text())


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(
        description="Demo server for Model 0 and Model 1 pipelines"
    )
    parser.add_argument(
        "--system_prompt_model1", type=str, help="Path to Model 1 system prompt file "
    )
    parser.add_argument(
        "--system_prompt_model0", type=str, help="Path to Model 0 system prompt file "
    )
    args = parser.parse_args()

    # Store the system prompt path in environment for access in websocket handler
    os.environ["MODEL1_SYSTEM_PROMPT"] = args.system_prompt_model1
    os.environ["MODEL0_SYSTEM_PROMPT"] = args.system_prompt_model0

    uvicorn.run(app)
