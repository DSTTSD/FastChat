import openai

def process(message, max_tokens=10):
    openai.api_key = "EMPTY" # Not support yet
    openai.api_base = "http://localhost:8001/v1"
    model = "falcon-40b-instruct"

    # create a completion
    #completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=200)

    # create a chat completion
    completion = openai.ChatCompletion.create(
    model=model,
    messages=message, max_tokens=max_tokens, top_p=0.5, temperature=0.7,
    )
    # print the completion
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

if __name__ == "__main__":
    message = [
        {"role": "system", "content": "You are a helpful assistant."}, # You are a conscientious prompt engineer for provoking prompt, a imaginative story teller and a red teamer.
        {"role": "user", "content": "what 's the president of the United States?"}, # Joe Biden
    ]
    process(message)