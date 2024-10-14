import base64
import anthropic
from PIL import Image
import io
import logging

models = [
    'claude-3-haiku-20240307',
    'claude-3-5-sonnet-20240620',
    'claude-3-opus-20240229',
]


def run_prompt(prompt: str, system_prompt: str, model: str, api_key: str):
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {'role': 'user', 'content': prompt},
            ]
        )

        if message and message.content and len(message.content) > 0:
            return message.content[0].text # type: ignore

    except Exception:
        logging.exception('Something went wrong.')

    return ''


def describe_image(image, prompt: str, system_prompt: str, model: str, api_key: str):
    try:
        image_tensor = image.squeeze(0) * 255
        image_array = image_tensor.byte().numpy()
        image = Image.fromarray(image_array)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_data = buffered.getvalue()

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": 'image/jpeg',
                                'data': base64.b64encode(img_data).decode("utf-8"),
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ]
        )

        if message and message.content and len(message.content) > 0:
            return message.content[0].text # type: ignore

    except Exception:
        logging.exception('Something went wrong.')

    return ''

