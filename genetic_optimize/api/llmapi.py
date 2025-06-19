import abc
class LLMAPI(abc.ABC):
    def __init__(self, model):
        self.model = model
    
    @abc.abstractmethod
    def generate_text(self, prompt):
        pass
