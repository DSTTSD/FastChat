export CUDA_VISIBLE_DEVICES="0,1"
controller_port=21001
model_port=21002
openai_port=8001
python3 -m fastchat.serve.controller --port $controller_port 
