class BaseGenerator:
    """
    Abstract base class for a generator model.
    """
    def generate(self, prompt: str) -> str:
        """
        Takes a full prompt and returns a string answer.
        """
        raise NotImplementedError