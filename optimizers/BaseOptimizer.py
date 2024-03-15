from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def optimize(self):
        pass

    @abstractmethod
    def get_updated_points(self):
        pass
