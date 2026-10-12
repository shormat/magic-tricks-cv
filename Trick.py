from abc import ABC, abstractmethod

class Trick(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def trick_1(self):
        raise NotImplementedError("Subclass must implement this method")
    
    @abstractmethod
    def trick_2(self):
        raise NotImplementedError("Subclass must implement this method")
    
    @abstractmethod
    def trick_3(self):
        raise NotImplementedError("Subclass must implement this method")
    