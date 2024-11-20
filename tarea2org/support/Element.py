# from app import *

import pyglet
import pyglet.gl as GL
import grafica.transformations as tr
import trimesh as tm
import pymunk
from grafica.textures import texture_2D_setup

from tarea2org.support.Vertex import Vertex

class Element(object):
    
    def __init__(self,
                 pipeline: pyglet.graphics.shader.ShaderProgram,
                 mesh_path,
                 transform = tr.identity(),
                 parent = None,
                ):
        self.__parent = parent
        self.pipeline = pipeline
        self.mesh_path = mesh_path
        self.mesh = tm.load(mesh_path)
        self.transform(transform)
        self.set_gpu()
        # GL
        self.root_set()
        self.body:pymunk.Poly  = None
        self.shape:pymunk.Poly = None
        self.transform_func    = lambda e: e.root
        # Set others
    
    def root_set(self):
        """  """
        parent: Element = self.__parent
        self.root = parent.root.copy() if parent else tr.identity()
    
    def draw(self, force=False, **data):
        self.pipeline["transform"] = (self.matrix_draw().reshape(16, 1, order="F"))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.gpu.texture)
        for attr in ["Kd", "Ks", "Ka"]: self.pipeline[attr] = self.gpu.__dict__[attr][:3]
        self.pipeline["shininess"] = self.gpu.ns
        
        if force:
            for attr, val in data.items():
                self.pipeline[attr] = val
        
        self.gpu.draw(pyglet.gl.GL_TRIANGLES)
        
    def matrix_draw(self):
        """ Return a mod of self.root """
        return self.transform_func(self)
        
    def transform(self, transformation): 
        self.mesh.apply_transform(transformation)
        self.vertex = Vertex(self.mesh)
        self.gpu = self.pipeline.vertex_list_indexed(
            self.vertex.number,
            GL.GL_TRIANGLES,
            self.vertex.list[3],
        )
    
    def move_body(self, x, y):
        self.body.position = (x, y)

    def transform_root(self,
                       tx=0,ty=0,tz=0,
                       rx=0,ry=0,rz=0,
                       s=1,
                       ):
        """ Translation goes first """
        """ Only before physics """
        self.root = self.root @ tr.translate(tx, ty, tz)
        self.root = self.root @ tr.rotationX(rx)
        self.root = self.root @ tr.rotationY(ry)
        self.root = self.root @ tr.rotationZ(rz)
        self.root = self.root @ tr.uniformScale(s)
        
        if self.body:
            x,y = self.body.position
            a   = self.body.angle
            self.body.position = (x+tx, y+tz)
            self.body.angle    = a+ry
    
    def set_gpu(self, force= False, Ka=-1, Ks=-1, Kd=-1, ns=-1, ):
        self.gpu.position[:] = self.vertex.list[4][1]
        self.gpu.normal[:]   = self.vertex.list[5][1]
        self.gpu.uv[:]       = self.vertex.list[6][1]
        self.gpu.texture     = texture_2D_setup(self.mesh.visual.material.image)
        
        self.gpu.Ka = Ka / 255 if force and not (Ka is -1) else self.mesh.visual.material.__dict__["ambient"]  / 255
        self.gpu.Ks = Ks / 255 if force and not (Ks is -1) else self.mesh.visual.material.__dict__["diffuse"]  / 255
        self.gpu.Kd = Kd / 255 if force and not (Kd is -1) else self.mesh.visual.material.__dict__["specular"] / 255
        self.gpu.ns = ns       if force and not (ns is -1) else self.mesh.visual.material.glossiness           #/ 255