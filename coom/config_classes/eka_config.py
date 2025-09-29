from nemo.collections.llm import DeepSeekV3Config


class EKAConfig(DeepSeekV3Config):
    """
    Currently identical to NeMo's DeepSeekV3Config.
    Defined separately for modularity, to allow future changes or extensions.
    """
