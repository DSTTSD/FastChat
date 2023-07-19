controller_port=21001
model_port=21002
openai_port=8000
python3 -m fastchat.serve.openai_api_server --host localhost --port $openai_port --controller-address http://localhost:${controller_port}
