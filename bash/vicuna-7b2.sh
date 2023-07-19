controller_port=21003
model_port=21004
openai_port=8003
python3 -m fastchat.serve.openai_api_server --host localhost --port $openai_port --controller-address http://localhost:${controller_port}
