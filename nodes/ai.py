"""AI functions."""

import base64
import io
import logging

import anthropic
from PIL import Image

models = [
    'claude-3-haiku-20240307',
    'claude-3-5-sonnet-20241022',
    'claude-3-opus-20240229',
]


def run_prompt(
    prompt: str, system_prompt: str, model: str, api_key: str
) -> str:
    """Execute a simple prompt with Claude.

    Args:
        prompt (str): The prompt to execute.
        system_prompt (str): The system prompt to use.
        model (str): The model to use.
        api_key (str): The API key to use.

    Returns:
        str: The result of the prompt.
    """
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {'role': 'user', 'content': prompt},
            ],
        )

        if message and message.content and len(message.content) > 0:
            return message.content[0].text  # type: ignore  # noqa: PGH003

    except Exception:
        logging.exception('Something went wrong.')

    return ''


def describe_image(
    image: 'torch.Tensor',  # type: ignore[name-defined]  # noqa: F821
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
) -> str:
    """Send an image to Claude's vision API.

    Args:
        image (torch.Tensor): The image to describe.
        prompt (str): The prompt to use.
        system_prompt (str): The system prompt to use.
        model (str): The model to use.
        api_key (str): The API key to use.

    Returns:
        str: The result of the prompt.
    """
    try:
        image_tensor = image.squeeze(0) * 255
        image_array = image_tensor.byte().numpy()
        image = Image.fromarray(image_array)
        buffered = io.BytesIO()
        image.save(buffered, format='JPEG')
        img_data = buffered.getvalue()

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': 'image/jpeg',
                                'data': base64.b64encode(img_data).decode(
                                    'utf-8'
                                ),
                            },
                        },
                        {
                            'type': 'text',
                            'text': prompt,
                        },
                    ],
                }
            ],
        )

        if message and message.content and len(message.content) > 0:
            return message.content[0].text  # type: ignore  # noqa: PGH003

    except Exception:
        logging.exception('Something went wrong.')

    return ''
