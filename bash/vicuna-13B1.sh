controller_port=21001
model_port=21002
openai_port=8000
# vicuna-13b-v1.1 model_name=vicuna-33b # llama
#model_path=/scratch/llm/vicuna-33b-v1.3
python3 -m fastchat.serve.model_worker \
        --model-name 'vicuna-13b' \
        --model-path ~/llm/models/vicuna-13b \
        --controller-address http://localhost:${controller_port} \
        --worker-address "http://localhost:${model_port}" \
        --port $model_port \
        --num-gpus 1
