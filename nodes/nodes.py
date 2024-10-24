"""Nodes for the ComfyUI Claude extension."""

from comfyui_types import (
    ChoiceInput,
    ComfyUINode,
    ImageInput,
    StringInput,
    StringOutput,
)

from .ai import describe_image, models, run_prompt

DESCRIBE_IMAGE_PROMPT = 'Describe this image in detail.'
COMBINE_TEXTS_PROMPT = 'Combine the following two texts into one coherent \
prompt without redundancies.'


class DescribeImage(ComfyUINode):
    """Describe an image."""

    image = ImageInput()
    model = ChoiceInput(choices=models)
    api_key = StringInput()

    system_prompt = StringInput(required=False, multiline=True)
    prompt = StringInput(
        required=False, multiline=True, default=DESCRIBE_IMAGE_PROMPT
    )

    description = StringOutput()

    def excecute(
        self,
        image: 'torch.Tensor',  # type: ignore[name-defined]  # noqa: F821
        model: str,
        api_key: str,
        system_prompt: str,
        prompt: str,
    ) -> tuple[str, ...]:
        """Send an image to Claude's vision API.

        Args:
            image (torch.Tensor): The image to describe.
            model (str): The model to use.
            api_key (str): The API key to use.
            system_prompt (str): The system prompt to use.
            prompt (str): The prompt to use.

        Returns:
            str: The result of the prompt.
        """
        return (describe_image(image, prompt, system_prompt, model, api_key),)


class CombineTexts(ComfyUINode):
    """Combine two texts."""

    text_1 = StringInput(multiline=True)
    text_1_prefix = StringInput(default='1')
    text_2 = StringInput(multiline=True)
    text_2_prefix = StringInput(default='2')
    model = ChoiceInput(choices=models)
    api_key = StringInput()

    system_prompt = StringInput(required=False, multiline=True)
    prompt = StringInput(
        required=False, multiline=True, default=DESCRIBE_IMAGE_PROMPT
    )

    combined_texts = StringOutput()

    def excecute(
        self,
        text_1: str,
        text_1_prefix: str,
        text_2: str,
        text_2_prefix: str,
        model: str,
        api_key: str,
        system_prompt: str,
        prompt: str,
    ) -> tuple[str, ...]:
        """Combine two texts.

        Args:
            text_1 (str): The first text.
            text_1_prefix (str): The prefix for the first text.
            text_2 (str): The second text.
            text_2_prefix (str): The prefix for the second text.
            model (str): The model to use.
            api_key (str): The API key to use.
            system_prompt (str): The system prompt to use.
            prompt (str): The prompt to use.

        Returns:
            str: The result of the prompt.
        """
        full_prompt = (
            f'{prompt}\n{text_1_prefix} {text_1}\n{text_2_prefix} {text_2}'
        )
        return (run_prompt(full_prompt, system_prompt, model, api_key),)


class TransformText(ComfyUINode):
    """Transform text."""

    text = StringInput(multiline=True)
    model = ChoiceInput(choices=models)
    api_key = StringInput()

    system_prompt = StringInput(required=False, multiline=True)
    prompt = StringInput(
        required=False, multiline=True, default=DESCRIBE_IMAGE_PROMPT
    )

    transformed_text = StringOutput()

    def excecute(
        self,
        text: str,
        model: str,
        api_key: str,
        system_prompt: str,
        prompt: str,
    ) -> tuple[str, ...]:
        """Transform text.

        Args:
            text (str): The text to transform.
            model (str): The model to use.
            api_key (str): The API key to use.
            system_prompt (str): The system prompt to use.
            prompt (str): The prompt to use.

        Returns:
            str: The result of the prompt.
        """
        full_prompt = f'{prompt}\nText: {text}\n'
        return (run_prompt(full_prompt, system_prompt, model, api_key),)
