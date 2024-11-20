class Projection(object):
    
    def __init__(self):
        self.__types = dict()
        self.__current = ""
    
    def add(self, p_name, perspective):
        self.__types[p_name] = perspective

    def get(self):
        return self.__types[self.__current].reshape(
            16, 1, order="F"
        )

    def __setattr__(self, name, value):
        if name == 'current':
            self.__current = value
            print(f"Projection set to:\n{self.get()}")
        else:
            super().__setattr__(name, value)