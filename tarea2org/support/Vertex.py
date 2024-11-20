import trimesh as tm

class Vertex(object):
    
    def __init__(self, mesh):
        self.list = tm.rendering.mesh_to_vertexlist(mesh)
        self.number = len(self.list[4][1]) // 3