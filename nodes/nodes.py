from .ai import models, describe_image, run_prompt


DESCRIBE_IMAGE_PROMPT = 'Describe this image in detail.'
COMBINE_TEXTS_PROMPT = 'Combine the following two texts into one coherent \
prompt without redundancies.'


class DescribeImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'model': (models,),
                'api_key': ('STRING',),
            },
            'optional': {
                'system_prompt': ('STRING', {'multiline': True}),
                'prompt': ('STRING', {
                    'default': DESCRIBE_IMAGE_PROMPT,
                    'multiline': True
                }),
            }
        }

    RETURN_TYPES = ('STRING',)
    FUNCTION = 'excecute'
    OUTPUT_NODE = True

    def excecute(
            self,
            image,
            model: str,
            api_key: str,
            system_prompt: str,
            prompt: str
    ):
        return (describe_image(image, prompt, system_prompt, model, api_key),)


class CombineTexts:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'text_1': ('STRING', {'multiline': True}),
                'text_1_prefix': ('STRING', {'default': '1)'}),
                'text_2': ('STRING', {'multiline': True}),
                'text_2_prefix': ('STRING', {'default': '2)'}),
                'model': (models,),
                'api_key': ('STRING',),
            },
            'optional': {
                'system_prompt': ('STRING', {'multiline': True}),
                'prompt': ('STRING', {
                    'default': COMBINE_TEXTS_PROMPT,
                    'multiline': True,
                }),
            }
        }

    RETURN_TYPES = ('STRING',)
    FUNCTION = 'excecute'
    OUTPUT_NODE = True

    def excecute(
            self,
            text_1: str,
            text_1_prefix: str,
            text_2: str,
            text_2_prefix: str,
            model: str,
            api_key: str,
            system_prompt: str,
            prompt: str
    ):
        full_prompt = f'{prompt}\n{text_1_prefix} {text_1}\n{text_2_prefix} {text_2}'
        return (run_prompt(full_prompt, system_prompt, model, api_key),)


class TransformText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'text': ('STRING', {'multiline': True}),
                'model': (models,),
                'api_key': ('STRING',),
            },
            'optional': {
                'system_prompt': ('STRING', {'multiline': True}),
                'prompt': ('STRING', {
                    'multiline': True,
                }),
            }
        }

    RETURN_TYPES = ('STRING',)
    FUNCTION = 'excecute'
    OUTPUT_NODE = True

    def excecute(
            self,
            text: str,
            model: str,
            api_key: str,
            system_prompt: str,
            prompt: str
    ):
        full_prompt = f'{prompt}\nText: {text}\n'
        return (run_prompt(full_prompt, system_prompt, model, api_key),)
