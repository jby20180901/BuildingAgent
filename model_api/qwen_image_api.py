import numpy as np
import random
import os
import time
import threading
import queue
from multiprocessing import Process, Queue, Event
import atexit
import signal
from io import BytesIO
import base64
from pydantic import BaseModel
from fastapi import FastAPI
from PIL import Image

import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

from diffusers import DiffusionPipeline
# from tools.prompt_utils import rewrite  # 你的 prompt 优化工具

# ----------------- 配置 -----------------
model_repo_id = "/home/jiangbaoyang/HuggingFace-Download-Accelerator/hf_hub/Qwen-Image" # 下载模型的存放路径
MAX_SEED = np.iinfo(np.int32).max

NUM_GPUS_TO_USE = int(os.environ.get("NUM_GPUS_TO_USE", 1))#torch.cuda.device_count()
TASK_QUEUE_SIZE = int(os.environ.get("TASK_QUEUE_SIZE", 100))
TASK_TIMEOUT = int(os.environ.get("TASK_TIMEOUT", 300))

print(f"Config: Using {NUM_GPUS_TO_USE} GPUs, queue size {TASK_QUEUE_SIZE}, timeout {TASK_TIMEOUT} seconds")

# ----------------- GPU Worker -----------------
class GPUWorker:
    def __init__(self, gpu_id, model_repo_id, task_queue, result_queue, stop_event):
        self.gpu_id = gpu_id
        self.model_repo_id = model_repo_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.device = f"cuda:{gpu_id}"
        self.pipe = None

    def initialize_model(self):
        try:
            torch.cuda.set_device(self.gpu_id)
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.pipe = DiffusionPipeline.from_pretrained(self.model_repo_id, torch_dtype=torch_dtype)
            self.pipe = self.pipe.to(self.device)
            print(f"GPU {self.gpu_id} model initialized successfully")
            return True
        except Exception as e:
            print(f"GPU {self.gpu_id} model initialization failed: {e}")
            return False

    def process_task(self, task):
        try:
            task_id = task['task_id']
            generator = torch.Generator(device=self.device).manual_seed(task['seed'])
            with torch.cuda.device(self.gpu_id):
                image = self.pipe(
                    prompt=task['prompt'],
                    negative_prompt=task['negative_prompt'],
                    true_cfg_scale=task['guidance_scale'],
                    num_inference_steps=task['num_inference_steps'],
                    width=task['width'],
                    height=task['height'],
                    generator=generator
                ).images[0]
            return {'task_id': task_id, 'image': image, 'success': True, 'gpu_id': self.gpu_id}
        except Exception as e:
            return {'task_id': task_id, 'success': False, 'error': str(e), 'gpu_id': self.gpu_id}

    def run(self):
        if not self.initialize_model():
            return
        print(f"GPU {self.gpu_id} worker starting")
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:
                    break
                result = self.process_task(task)
                self.result_queue.put(result)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"GPU {self.gpu_id} worker exception: {e}")
        print(f"GPU {self.gpu_id} worker stopping")

def gpu_worker_process(gpu_id, model_repo_id, task_queue, result_queue, stop_event):
    worker = GPUWorker(gpu_id, model_repo_id, task_queue, result_queue, stop_event)
    worker.run()

# ----------------- 多 GPU 管理 -----------------
class MultiGPUManager:
    def __init__(self, model_repo_id, num_gpus=None, task_queue_size=100):
        self.model_repo_id = model_repo_id
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.task_queue = Queue(maxsize=task_queue_size)
        self.result_queue = Queue()
        self.stop_event = Event()
        self.worker_processes = []
        self.task_counter = 0
        self.pending_tasks = {}

    def start_workers(self):
        for gpu_id in range(self.num_gpus):
            process = Process(target=gpu_worker_process,
                              args=(gpu_id, self.model_repo_id, self.task_queue,
                                    self.result_queue, self.stop_event))
            process.start()
            self.worker_processes.append(process)
        threading.Thread(target=self._process_results, daemon=True).start()
        print(f"All {self.num_gpus} GPU workers have started")

    def _process_results(self):
        while not self.stop_event.is_set():
            try:
                result = self.result_queue.get(timeout=1)
                task_id = result['task_id']
                if task_id in self.pending_tasks:
                    self.pending_tasks[task_id]['result'] = result
                    self.pending_tasks[task_id]['event'].set()
            except queue.Empty:
                continue

    def submit_task(self, prompt, negative_prompt="", seed=42, width=1664, height=928,
                    guidance_scale=4, num_inference_steps=50, timeout=300):
        task_id = f"task_{self.task_counter}_{time.time()}"
        self.task_counter += 1
        task = {
            'task_id': task_id, 'prompt': prompt, 'negative_prompt': negative_prompt,
            'seed': seed, 'width': width, 'height': height,
            'guidance_scale': guidance_scale, 'num_inference_steps': num_inference_steps
        }
        result_event = threading.Event()
        self.pending_tasks[task_id] = {'event': result_event, 'result': None}
        self.task_queue.put(task, timeout=10)

        start_time = time.time()
        while not result_event.is_set():
            if result_event.wait(timeout=2):
                break
            if time.time() - start_time > timeout:
                del self.pending_tasks[task_id]
                return {'success': False, 'error': 'Task timeout'}
        result = self.pending_tasks[task_id]['result']
        del self.pending_tasks[task_id]
        return result

    def stop(self):
        print("Stopping Multi-GPU Manager...")
        self.stop_event.set()
        for _ in range(self.num_gpus):
            self.task_queue.put(None)
        for process in self.worker_processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        print("Multi-GPU Manager stopped")

gpu_manager = None
def initialize_gpu_manager():
    global gpu_manager
    if gpu_manager is None:
        gpu_manager = MultiGPUManager(model_repo_id, num_gpus=NUM_GPUS_TO_USE, task_queue_size=TASK_QUEUE_SIZE)
        gpu_manager.start_workers()
        print("GPU Manager initialized successfully")

# ----------------- 推理接口 -----------------
def get_image_size(aspect_ratio):
    if aspect_ratio == "1:1": return 1328, 1328
    elif aspect_ratio == "16:9": return 1664, 928
    elif aspect_ratio == "9:16": return 928, 1664
    elif aspect_ratio == "4:3": return 1472, 1140
    elif aspect_ratio == "3:4": return 1140, 1472
    return 1328, 1328

def infer(prompt, negative_prompt="", seed=42, randomize_seed=False,
          aspect_ratio="16:9", guidance_scale=5, num_inference_steps=50):
    global gpu_manager
    if gpu_manager is None:
        initialize_gpu_manager()
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    width, height = get_image_size(aspect_ratio)
    # prompt = rewrite(prompt)
    result = gpu_manager.submit_task(prompt, negative_prompt, seed, width, height,
                                     guidance_scale, num_inference_steps, timeout=TASK_TIMEOUT)
    return result['image'], seed if result['success'] else (None, seed)

# ----------------- FastAPI -----------------
app = FastAPI()

class InferenceRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    seed: int = 42
    randomize_seed: bool = False
    aspect_ratio: str = "16:9"
    guidance_scale: float = 5.0
    num_inference_steps: int = 50

@app.post("/generate")
def generate_image(req: InferenceRequest):
    image, seed = infer(req.prompt, req.negative_prompt, req.seed, req.randomize_seed,
                        req.aspect_ratio, req.guidance_scale, req.num_inference_steps)
    if image is None:
        return {"success": False, "error": "Generation failed"}
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return {"success": True, "seed": seed, "image_base64": img_str}

# ----------------- 退出清理 -----------------
def cleanup():
    if gpu_manager:
        gpu_manager.stop()
atexit.register(cleanup)
signal.signal(signal.SIGINT, lambda s, f: cleanup())
signal.signal(signal.SIGTERM, lambda s, f: cleanup())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8021)
