from nemo.collections.llm import DeepSeekModel


class EKAModel(DeepSeekModel):
    """
    Currently identical to NeMo's DeepSeekModel.
    Defined separately for modularity, to allow future changes or extensions.
    """
