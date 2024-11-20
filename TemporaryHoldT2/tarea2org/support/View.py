import grafica.transformations as tr
import numpy as np

class View(object):
    
    def __init__(self):
        self.look_from  = None
        self.look_to    = None
        self.projection = None
        self.__views    = dict()
        self.__keys     = list()
        self.__current  = None
        
    def add(self, name, look_from, look_to, projection):
        """ Remember: projection is a str reference to a Projection """
        self.__views[name] = (
            look_from,
            look_to,
            projection,
        )
        self.__keys = list(self.__views.keys())
        self.set_to(name)
    
    def set_to(self, name):
        self.__current  = name
        self.look_from  = self.__views[name][0]
        self.look_to    = self.__views[name][1]
        self.projection = self.__views[name][2]
        print("Set to:", self.projection)
        
    def matrix(self):
        return tr.lookAt(
            np.array(self.look_from),
            np.array(self.look_to),
            np.array([0.0, 1.0, 0.0])
        )
    
    def vec3from(self):
        return self.look_from

    def next(self):
        ln   = len(self.__keys)
        id   = self.__keys.index(self.__current)
        next = self.__keys[(id + 1) % ln]
        
        self.set_to(next)