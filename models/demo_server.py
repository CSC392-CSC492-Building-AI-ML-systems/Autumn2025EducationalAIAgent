#!/usr/bin/env python3
"""
FastAPI server for real-time visualization of Model 0 and Model 1 pipelines.
Streams processing updates via WebSocket.
"""

import asyncio
import os
import re
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

app = FastAPI()

# Global state
current_group_events: List[str] = []
current_group_id: int = 0
model0_instance = None
processing_complete = False
processing_in_progress = False

# Store all messages for replay to new clients
message_history: List[Dict] = []

# Store all connected WebSocket clients
connected_clients: List[WebSocket] = []

sortme_pattern = re.compile(
    r"<(?:user_input|system_output)\b[^>]*?sortme=\"True\"[^>]*?(?:/>|>.*?</(?:user_input|system_output)>)",
    flags=re.DOTALL,
)


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
                time.sleep(2 ** attempt)  # Exponential backoff
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print("Rate limit exceeded. Waiting before retry...")
                time.sleep(5)
            elif e.response.status_code >= 500:
                print(f"Server error: {e.response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            else:
                raise
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    raise Exception("Max retries exceeded")


class Model0:
    def __init__(self):
        print("Initializing Model 0 vLLM client...")
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")


        self.base_url = os.environ.get("MODEL0_BASE_URL")
        self.api_key = os.environ.get("MODEL0_API_KEY")
        
        self.url = f"{self.base_url}/v1/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        print(f"Model 0 configured for endpoint: {self.base_url}")

    def generate(self, prompt: str) -> str:
        
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        payload = {
            "model": "educational",
            "prompt": formatted_prompt,
            "temperature": 0.5,
            "max_tokens": 512
        }

        print(f"Calling Model 0 API with prompt length: {len(prompt)}")

        try:
            result = send_vllm_request(self.url, self.headers, payload)

            if "choices" in result and len(result["choices"]) > 0:
                generated_text = result["choices"][0]["text"]
                generated_text = generated_text.replace('</think>', '').strip()
                print(f"Model 0 response received: {len(generated_text)} chars")
                return generated_text.strip()
            else:
                print(f"Unexpected response format: {result}")
                return "ERROR: Unexpected response format"

        except Exception as e:
            error_msg = f"Model 0 generation failed: {str(e)}"
            print(error_msg)
            return f"ERROR: {error_msg}"


def extract_clean_event(event_with_attrs: str) -> str:
    return re.sub(r'\s+(group|sortme)="[^"]*"', '', event_with_attrs)


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
    """Process a single group with Model 1 in the background"""
    try:
        # Show we're processing this group
        await broadcast_message({
            "type": "model1_input",
            "data": {
                "group_id": group_id,
                "events": events
            }
        })

        # Simulate Model 1 API call (replace with actual RunPod API call)
        await asyncio.sleep(2.0)  # Simulates slow API response
        summary = f"Summary of group {group_id} with {len(events)} events"

        # Send result when it arrives
        await broadcast_message({
            "type": "model1_summary",
            "data": {
                "group_id": group_id,
                "summary": summary
            }
        })

    except Exception as e:
        print(f"Error in Model 1 processing: {e}")


async def process_model0_pipeline(dataset, system_prompt: str):
    """Process Model 0 pipeline - continues while Model 1 runs in background"""
    global current_group_events, current_group_id, model0_instance, processing_complete, processing_in_progress

    if processing_in_progress or processing_complete:
        return

    processing_in_progress = True

    model0_instance = Model0()

    current_group_events = []
    current_group_id = 0

    for idx, example in enumerate(dataset['train']):
        input_xml = example['input']
        expected_output = example['output']

        # Extract the new event
        sortme_match = sortme_pattern.search(input_xml)
        if not sortme_match:
            continue

        sortme_event = sortme_match.group(0)
        clean_event = extract_clean_event(sortme_event)

        await broadcast_message({
            "type": "raw_event",
            "data": clean_event
        })

        await broadcast_message({
            "type": "model0_input",
            "data": {
                "input_xml": input_xml,
                "current_event": clean_event,
                "group_id": current_group_id
            }
        })

        # Generate Model 0 response
        prompt = f"{system_prompt}\n\n{input_xml}"
        response = model0_instance.generate(prompt)

        # Send Model 0 full response for visualization
        await broadcast_message({
            "type": "model0_thinking",
            "data": {
                "full_response": response
            }
        })

        prediction_type = parse_model0_response(expected_output)

        # Send Model 0 answer for visualization
        await broadcast_message({
            "type": "model0_answer",
            "data": {
                "answer": prediction_type,
                "group_id": current_group_id
            }
        })

        if prediction_type == "NEW":
            if current_group_events:
                asyncio.create_task(process_model1(current_group_id, current_group_events.copy()))

            current_group_id += 1
            current_group_events = [clean_event]
        else:
            current_group_events.append(clean_event)

        await broadcast_message({
            "type": "current_group_update",
            "data": {
                "group_id": current_group_id,
                "events": current_group_events
            }
        })

        await asyncio.sleep(0.3)

    if current_group_events:
        asyncio.create_task(process_model1(current_group_id, current_group_events.copy()))

    await broadcast_message({"type": "complete"})
    processing_complete = True
    processing_in_progress = False


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming updates"""
    global connected_clients, message_history, processing_complete, processing_in_progress

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
                "patea4/educational-ai-agent-small",
                data_files="1721655754.jsonl"
            )

            system_prompt_path = Path(__file__).parent / "model_0" / "system_prompt.txt"
            system_prompt = system_prompt_path.read_text()

            asyncio.create_task(process_model0_pipeline(dataset, system_prompt))

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
            await websocket.send_json({
                "type": "error",
                "data": str(e)
            })
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
