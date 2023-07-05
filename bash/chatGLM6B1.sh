controller_port=21003
model_port=21004
openai_port=8003
# vicuna-13b-v1.1 model_name=vicuna-33b # llama
#model_path=/scratch/llm/vicuna-33b-v1.3
python3 -m fastchat.serve.model_worker --model-name 'chatGLM' --model-path /scratch/llm/chatglm-6b --controller-address http://localhost:${controller_port} --port $model_port 