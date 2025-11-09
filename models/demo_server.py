#!/usr/bin/env python3
"""
FastAPI server for real-time visualization of Model 0 and Model 1 pipelines.
Streams processing updates via WebSocket.
"""

import asyncio
import json
import re
from typing import List, Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import torch
from peft import AutoPeftModelForCausalLM
from datasets import load_dataset
from transformers import AutoTokenizer

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


class Model0:
    def __init__(self):
        print("Loading Model 0...")
        adapter_id = "patea4/deepseek-r1-educational-lora-tuned3"

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_id, use_fast=True)
        self.tokenizer.truncation_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoPeftModelForCausalLM.from_pretrained(
            adapter_id,
            device_map="cuda:0",
        )
        print("Model 0 loaded!")

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.5,
                do_sample=True,
            )

        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()


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


async def process_model0_pipeline(dataset, system_prompt: str):
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
                # Send completed group to Model 1 Visualization
                await broadcast_message({
                    "type": "model1_input",
                    "data": {
                        "group_id": current_group_id,
                        "events": current_group_events
                    }
                })

                # Simulate Model 1 processing
                await asyncio.sleep(0.5)
                summary = f"Summary of group {current_group_id} with {len(current_group_events)} events"

                # Send model 1 output to visualization
                await broadcast_message({
                    "type": "model1_summary",
                    "data": {
                        "group_id": current_group_id,
                        "summary": summary
                    }
                })

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
        await broadcast_message({
            "type": "model1_input",
            "data": {
                "group_id": current_group_id,
                "events": current_group_events
            }
        })

        summary = f"Summary of group {current_group_id} with {len(current_group_events)} events"
        await broadcast_message({
            "type": "model1_summary",
            "data": {
                "group_id": current_group_id,
                "summary": summary
            }
        })

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

            # Start processing (runs in background)
            asyncio.create_task(process_model0_pipeline(dataset, system_prompt))

        while True:
            try:
                # Wait for messages from client (keep alive)
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
