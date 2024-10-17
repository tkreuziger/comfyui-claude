from .nodes.nodes import CombineTexts, DescribeImage, TransformText

NODE_CLASS_MAPPINGS = {
    'Describe Image': DescribeImage,
    'Combine Texts': CombineTexts,
    'Transform Text': TransformText,
}

__all__ = ['NODE_CLASS_MAPPINGS']
