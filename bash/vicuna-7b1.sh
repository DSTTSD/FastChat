controller_port=21003
model_port=21004
openai_port=8003
# vicuna-13b-v1.1 model_name=vicuna-33b # llama
#model_path=/scratch/llm/vicuna-33b-v1.3
python3 -m fastchat.serve.model_worker --model-name 'vicuna-7b-v1.3' \
    --model-path lmsys/vicuna-7b-v1.3 \
    --controller-address http://localhost:${controller_port} \
    --worker-address "http://localhost:${model_port}" \
    --port $model_port 