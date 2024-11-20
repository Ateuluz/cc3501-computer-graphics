import grafica.transformations as tr

from elements.MyObject import MyObject

class MyParticle(MyObject):
    
    def __init__(self,
                 program_state,
                 pipeline,
                 mesh_path,
                 position,
                 span,
                 transform = tr.identity()
                 ):
        super().__init__(
            program_state,
            pipeline,
            mesh_path,
            position,
            transform,
        )
        self.born = program_state["total_time"]
        self.span = span
        self.size = 1
        self.static = True
        self.data = dict()