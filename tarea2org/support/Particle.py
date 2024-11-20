import pyglet
import pyglet.gl as GL
import grafica.transformations as tr
import trimesh as tm
import pymunk
from grafica.textures import texture_2D_setup

from tarea2org.support.Vertex import Vertex

from Element import Element

class Particle(Element):
    
    def __init__(self,
                 pipeline: pyglet.graphics.shader.ShaderProgram,
                 mesh_path,
                 pos_func,
                 program_state,
                 transform = tr.identity(),
                 parent = None,
                ):
        super().__init__(self, pipeline, mesh_path, transform, parent,)
        self.data = dict()
        self.pos_func = pos_func
        self.program_state = program_state
    
    def matrix_draw(self):
        """ Return a mod of self.root """
        return self.pos_func(self,self.program_state["total_time"])