#'guanaco-33b-merged'
# /data/private/shitongduan/llm/guanaco-33b-merged
export CUDA_VISIBLE_DEVICES="0,1,2,3"
num_gpus=4
controller_port=21001
model_port=21002
openai_port=8001
model_name=falcon-40b-instruct
model_path=/data/private/shitongduan/llm/falcon-40b-instruct
python3 -m fastchat.serve.model_worker --model-name $model_name --model-path $model_path --controller-address http://localhost:${controller_port} --port $model_port --worker-address http://localhost:${model_port} --num-gpus ${num_gpus} 