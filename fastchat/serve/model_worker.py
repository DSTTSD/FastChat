"""
A model worker executes the model.
"""
import argparse
import asyncio
import dataclasses
import logging
import json
import os
import time
from typing import List, Union
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import requests

from fastchat.modules.gptq import GptqConfig

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LlamaTokenizer,
        AutoModel,
    )
except ImportError:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LLaMATokenizer,
        AutoModel,
    )
import torch
import torch.nn.functional as F
import uvicorn

from fastchat.constants import WORKER_HEART_BEAT_INTERVAL, ErrorCode, SERVER_ERROR_MSG
from fastchat.model.model_adapter import (
    load_model,
    add_model_args,
    get_conversation_template,
)
from fastchat.model.chatglm_model import chatglm_generate_stream
from fastchat.model.falcon_model import falcon_generate_stream
from fastchat.serve.inference import generate_stream
from fastchat.utils import build_logger, pretty_print_semaphore

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class ModelWorker:
    def __init__(
        self,
        controller_addr,
        worker_addr,
        worker_id,
        no_register,
        model_path,
        model_names,
        device,
        num_gpus,
        max_gpu_memory,
        load_8bit=False,
        cpu_offloading=False,
        gptq_config=None,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        self.model_names = model_names or [model_path.split("/")[-1]]
        self.device = device

        logger.info(f"Loading the model {self.model_names} on worker {worker_id} ...")
        self.model, self.tokenizer = load_model(
            model_path,
            device,
            num_gpus,
            max_gpu_memory,
            load_8bit,
            cpu_offloading,
            gptq_config,
        )
        self.conv = get_conversation_template(model_path)
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if hasattr(self.model.config, "max_sequence_length"):
            self.context_len = self.model.config.max_sequence_length
        elif hasattr(self.model.config, "seq_length"):
            self.context_len = self.model.config.seq_length
        elif hasattr(self.model.config, "max_position_embeddings"):
            self.context_len = self.model.config.max_position_embeddings
        elif hasattr(self.model.config, "seq_length"):
            self.context_len = self.model.config.seq_length
        else:
            self.context_len = 2048

        # generate_stream
        is_chatglm = "chatglm" in str(type(self.model)).lower()
        is_falcon = "rwforcausallm" in str(type(self.model)).lower()
        if is_chatglm:
            self.generate_stream_func = chatglm_generate_stream
        elif is_falcon:
            self.generate_stream_func = falcon_generate_stream
        else:
            self.generate_stream_func = generate_stream

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,)
            )
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status(),
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(
            f"Send heart beat. Models: {self.model_names}. "
            f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
            f"global_counter: {global_counter}. "
            f"worker_id: {worker_id}. "
        )

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(
                    url,
                    json={
                        "worker_name": self.worker_addr,
                        "queue_length": self.get_queue_length(),
                    },
                    timeout=5,
                )
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if (
            model_semaphore is None
            or model_semaphore._value is None
            or model_semaphore._waiters is None
        ):
            return 0
        else:
            return (
                args.limit_model_concurrency
                - model_semaphore._value
                + len(model_semaphore._waiters)
            )

    def get_status(self):
        return {
            "model_names": self.model_names,
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    def count_token(self, params):
        prompt = params["prompt"]
        input_ids = self.tokenizer(prompt).input_ids
        input_echo_len = len(input_ids)

        ret = {
            "count": input_echo_len,
            "error_code": 0,
        }
        return ret

    def get_conv_template(self):
        return {"conv": self.conv}

    def generate_stream_gate(self, params):
        try:
            for output in self.generate_stream_func(
                self.model,
                self.tokenizer,
                params,
                self.device,
                self.context_len,
                args.stream_interval,
            ):
                ret = {
                    "text": output["text"],
                    "error_code": 0,
                }
                if "usage" in output:
                    ret["usage"] = output["usage"]
                if "finish_reason" in output:
                    ret["finish_reason"] = output["finish_reason"]
                if "logprobs" in output:
                    ret["logprobs"] = output["logprobs"]
                yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield json.dumps(ret).encode() + b"\0"
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield json.dumps(ret).encode() + b"\0"

    def generate_gate(self, params):
        try:
            ret = {"text": "", "error_code": 0}
            for output in self.generate_stream_func(
                self.model,
                self.tokenizer,
                params,
                self.device,
                self.context_len,
                args.stream_interval,
            ):
                ret["text"] = output["text"]
                if "usage" in output:
                    ret["usage"] = output["usage"]
                if "finish_reason" in output:
                    ret["finish_reason"] = output["finish_reason"]
                if "logprobs" in output:
                    ret["logprobs"] = output["logprobs"]
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret

    @torch.inference_mode()
    def get_embeddings(self, params):
        try:
            tokenizer = self.tokenizer
            is_llama = "llama" in str(
                type(self.model)
            )  # vicuna support batch inference
            is_chatglm = "chatglm" in str(type(self.model))
            is_t5 = "t5" in str(type(self.model))
            if is_llama:
                encoding = tokenizer.batch_encode_plus(
                    params["input"], padding=True, return_tensors="pt"
                )
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                model_output = self.model(
                    input_ids, attention_mask, output_hidden_states=True
                )
                data = model_output.hidden_states[-1]
                mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
                masked_embeddings = data * mask
                sum_embeddings = torch.sum(masked_embeddings, dim=1)
                seq_length = torch.sum(mask, dim=1)
                embedding = sum_embeddings / seq_length
                normalized_embeddings = F.normalize(embedding, p=2, dim=1)
                ret = {
                    "embedding": normalized_embeddings.tolist(),
                    "token_num": torch.sum(attention_mask).item(),
                }
            else:
                embedding = []
                token_num = 0
                for text in params["input"]:
                    input_ids = tokenizer.encode(text, return_tensors="pt").to(
                        self.device
                    )
                    if is_t5:
                        model_output = self.model(
                            input_ids, decoder_input_ids=input_ids
                        )
                    else:
                        model_output = self.model(input_ids, output_hidden_states=True)
                    if is_chatglm:
                        data = (model_output.hidden_states[-1].transpose(0, 1))[0]
                    elif is_t5:
                        data = model_output.encoder_last_hidden_state[0]
                    else:
                        data = model_output.hidden_states[-1][0]
                    data = F.normalize(torch.mean(data, dim=0), p=2, dim=0)
                    embedding.append(data.tolist())
                    token_num += len(input_ids[0])
                ret = {
                    "embedding": embedding,
                    "token_num": token_num,
                }
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret
    
    @torch.inference_mode()
    def sentence_energy_plus(self, prefix, suffix, repetition_penalty=1.0):
        """
        incoporate the repitation penalty
        """
        tokenizer = self.tokenizer
        model = self.model
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        
        input_tokens = prefix_tokens + suffix_tokens
        input_tokens = torch.tensor([input_tokens]).to(model.device)
        with torch.no_grad():
            outputs = model(input_tokens)
            logits = outputs.logits
        for i, token in enumerate(prefix_tokens):
            if token in suffix_tokens:
                logits[0, i, token] /= repetition_penalty
        log_likelihood = torch.mean(torch.gather(logits, 2, input_tokens.unsqueeze(2)).squeeze(2))
        energy = -log_likelihood
        print(f"penalty:{repetition_penalty};  energy:{energy.item()}")
        return energy.item()

    @torch.inference_mode()
    def conditional_sentence_energy(self, prefix, suffix):
        """
        E(Z| theta)
        """
        tokenizer = self.tokenizer
        model = self.model
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False, return_tensors='pt')
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False, return_tensors='pt')
        prefix_tokens = prefix_tokens.to(model.device)
        suffix_tokens = suffix_tokens.to(model.device)
        input_tokens = torch.cat((prefix_tokens, suffix_tokens), dim=1)
        with torch.no_grad():
            outputs = model(input_tokens)
            logits = outputs.logits[0, -len(suffix_tokens):, :]
            log_probabilities = torch.log_softmax(logits, dim=-1)
            suffix_probability = log_probabilities[torch.arange(len(suffix_tokens)), suffix_tokens.squeeze()]
            energy = -suffix_probability.mean()
        # print("conditional energy:", energy)
        return energy.item()

    # Function to score a sentence
    @torch.inference_mode()
    def sentence_energy(self, text):
        """
        E(theta, Z)
        """
        tokenizer = self.tokenizer
        model = self.model

        inputs = tokenizer.encode_plus(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            # retrieves the logits from the model's output.(except the last one)
            logits = outputs.logits[:, :-1, :].contiguous()
            #  extracts the labels from the input IDs. 
            labels = input_ids[:, 1:].contiguous()
            # compute the log probabilities of the tokens
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            # select the log probability of the labels
            log_prob_mean = torch.gather(log_probs, dim=-1,index=labels.unsqueeze(-1)).squeeze(-1).mean(dim=1) # 
        energy = -log_prob_mean.mean().item()
        print("energy:", energy)
        return energy
    
    @torch.inference_mode()
    def conditional_sentence_score(self, prefix, suffix):
        """
        E(Z| theta)
        """
        tokenizer = self.tokenizer
        model = self.model

        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False, return_tensors='pt')
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False, return_tensors='pt')
        prefix_tokens = prefix_tokens.to(model.device)
        suffix_tokens = suffix_tokens.to(model.device)
        input_tokens = torch.cat((prefix_tokens, suffix_tokens), dim=1)
        with torch.no_grad():
            outputs = model(input_tokens)
            logits = outputs.logits[0, -len(suffix_tokens):, :]
            log_probabilities = torch.log_softmax(logits, dim=-1)
            suffix_probability = log_probabilities[torch.arange(len(suffix_tokens)), suffix_tokens.squeeze()]   
        return suffix_probability.mean().item()

    @torch.inference_mode()
    def score_sentence(self, text):
        """
        E(theta, Z)
        """
        tokenizer = self.tokenizer
        model = self.model

        inputs = tokenizer.encode_plus(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            # retrieves the logits from the model's output.(except the last one)
            logits = outputs.logits[:, :-1, :].contiguous()
            #  extracts the labels from the input IDs. 
            labels = input_ids[:, 1:].contiguous()
            # compute the log probabilities of the tokens
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            # select the log probability of the labels
            log_prob_mean = torch.gather(log_probs, dim=-1,index=labels.unsqueeze(-1)).squeeze(-1).mean(dim=1) # 
        score = log_prob_mean.mean().item()
        return score

app = FastAPI()


def release_model_semaphore():
    model_semaphore.release()


def acquire_model_semaphore():
    global model_semaphore, global_counter
    global_counter += 1
    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    return model_semaphore.acquire()


def create_background_tasks():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_model_semaphore()
    generator = worker.generate_stream_gate(params)
    background_tasks = create_background_tasks()
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_model_semaphore()
    output = worker.generate_gate(params)
    release_model_semaphore()
    return JSONResponse(output)


@app.post("/worker_generate_completion_stream")
async def api_generate_completion_stream(request: Request):
    params = await request.json()
    await acquire_model_semaphore()
    generator = worker.generate_stream_gate(params)
    background_tasks = create_background_tasks()
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate_completion")
async def api_generate_completion(request: Request):
    params = await request.json()
    await acquire_model_semaphore()
    completion = worker.generate_gate(params)
    background_tasks = create_background_tasks()
    return JSONResponse(content=completion, background=background_tasks)


@app.post("/worker_get_embeddings")
async def api_get_embeddings(request: Request):
    params = await request.json()
    await acquire_model_semaphore()
    embedding = worker.get_embeddings(params)
    background_tasks = create_background_tasks()
    return JSONResponse(content=embedding, background=background_tasks)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def model_details(request: Request):
    return {"context_length": worker.context_len}

# my energy part
@app.post('/conditional_energy')
async def conditional_sentence_energy_interface(data: dict):
    prefix = data['prefix']
    suffix = data['suffix']
    energy = worker.conditional_sentence_energy(prefix, suffix)
    print("conditional energy:", energy)
    return {'energy': energy}

@app.post('/energy')
async def sentence_energy_interface(data: dict):
    text = data['text']
    energy = worker.sentence_energy(text)
    return {'energy': energy}

@app.post('/score')
async def score_sentence_interface(data: dict):
    text = data['text']
    score = worker.score_sentence(text)
    return {'score': score}

@app.post('/conditional_score')
async def conditional_score_sentence_interface(data: dict):
    prefix = data['prefix']
    suffix = data['suffix']
    score = worker.conditional_sentence_score(prefix, suffix)
    return {'score': score}

@app.post('/energy_plus')
async def sentence_energy_plus_interface(data: dict):
    prefix = data['prefix']
    suffix = data['suffix']
    repitition_penalty = data['repetition_penalty']
    energy = worker.sentence_energy_plus(prefix, suffix, repitition_penalty)
    return {'energy': energy}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    add_model_args(parser)
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    gptq_config = GptqConfig(
        ckpt=args.gptq_ckpt or args.model_path,
        wbits=args.gptq_wbits,
        groupsize=args.gptq_groupsize,
        act_order=args.gptq_act_order,
    )

    worker = ModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.no_register,
        args.model_path,
        args.model_names,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
        gptq_config,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
