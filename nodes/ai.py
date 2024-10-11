import base64
import anthropic
import torch
from torchvision import transforms

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
        print('Something went wrong.')

    return ''


def describe_image(image, prompt: str, system_prompt: str, model: str, api_key: str):
    try:
        transform = transforms.ToPILImage()
        pil_image = transform(torch.squeeze(image))
        img_data = pil_image.tobytes()

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {'role': 'user', 'content': [
                    { # type: ignore
                        'type': 'base64',
                        'media_type': 'image/jpeg',
                        'data': base64.b64encode(img_data).decode("utf-8"),
                    },
                    {
                        'type': 'text',
                        'text': prompt,
                    }
                ]},
            ]
        )

        if message and message.content and len(message.content) > 0:
            return message.content[0].text # type: ignore

    except Exception:
        print('Something went wrong.')

    return ''

