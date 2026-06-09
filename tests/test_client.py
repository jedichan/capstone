import os
from openai import OpenAI

# Initialize the OpenAI client pointing to the local vLLM server
client = OpenAI(
    api_key="EMPTY", # vLLM doesn't require an API key by default
    base_url="http://localhost:8000/v1",
)

# Example: Chat with an image
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"
                    },
                },
            ],
        }
    ],
)

print(response.choices[0].message.content)
