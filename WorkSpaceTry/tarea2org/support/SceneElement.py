import pyglet
import pyglet.gl as GL
import trimesh as tm
import grafica.transformations as tr

import numpy as np

from grafica.textures import texture_2D_setup

from tarea2org.support.Vertex import Vertex

class SceneElement(object):
    
    def __init__(self,
                 pipeline: pyglet.graphics.shader.ShaderProgram,
                 mesh_path,
                 transform = tr.identity(),
                 parent = None
                ):
        self.__parent = parent
        self.pipeline = pipeline
        self.mesh_path = mesh_path
        self.asset = tm.load(mesh_path)
        self.components = {}
        self.vertexs: dict[Vertex] = {}
        self.transform(transform)
        self.__set_mesh()
        self.root_set()
        self.transform_func = lambda e: e.root
    
    def root_set(self):
        parent: SceneElement = self.__parent
        self.root = parent.root.copy() if parent else tr.identity()
    
    def transform(self, transformation): 
        for comp in self.components.values():
            
            #mesh = comp["mesh"]
            
            comp["mesh"].apply_transform(transformation)
        
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
        
        # if self.body:
        #     x,y = self.body.position
        #     a   = self.body.angle
        #     self.body.position = (x+tx, y+tz)
        #     self.body.angle    = a+ry
    
    def matrix_draw(self):
        """ Return a mod of self.root """
        return self.transform_func(self)
    
    def __set_gpu(self, comp):
        
        vertex: Vertex = comp["vertex"]
        
        comp["gpu"] = self.pipeline.vertex_list_indexed(
            vertex.number, GL.GL_TRIANGLES, vertex.list[3])
        gpu = comp["gpu"]

        gpu.position[:] = vertex.list[4][1]
        gpu.normal[:]   = vertex.list[5][1]
        #gpu.uv[:]       = vertex.list[6][1]
        try:
            gpu.texture = texture_2D_setup(comp["mesh"].visual.material.image)
            
            comp["gpu"].Ka = comp["mesh"].visual.material.__dict__["ambient"]  / 255
            comp["gpu"].Ks = comp["mesh"].visual.material.__dict__["diffuse"]  / 255
            comp["gpu"].Kd = comp["mesh"].visual.material.__dict__["specular"] / 255
            comp["gpu"].ns = comp["mesh"].visual.material.glossiness           / 255
        except:
            #gpu.texture = texture_2D_setup(comp["mesh"].visual.material.image)
            
            comp["gpu"].Ka = np.array([.2,.2,.2]) #comp["mesh"].visual.material.__dict__["ambient"]  / 255
            comp["gpu"].Ks = np.array([.2,.2,.2]) #comp["mesh"].visual.material.__dict__["diffuse"]  / 255
            comp["gpu"].Kd = np.array([.2,.2,.2]) #comp["mesh"].visual.material.__dict__["specular"] / 255
            comp["gpu"].ns = .2 #comp["mesh"].visual.material.glossiness           / 255
            
    
    def __set_mesh(self):
        for object_id, object_geometry in self.asset.geometry.items():
            
            self.components[object_id] = {}
            comp = self.components[object_id]
            
            object_geometry.fix_normals(True)
            
            comp["mesh"] = object_geometry
            
            comp["vertex"] = Vertex(comp["mesh"])
            
            self.__set_gpu(comp)
    
    def draw(self, force=False, **data) -> None: # on_draw
        for comp in self.components.values():
            
            self.pipeline["transform"] = (self.matrix_draw().reshape(16, 1, order="F"))

            try:
                GL.glBindTexture(GL.GL_TEXTURE_2D, comp["texture"])
                
                for attr in ["Kd", "Ks", "Ka"]: self.pipeline[attr] = comp["gpu"].__dict__[attr][:3]
                self.pipeline["shininess"] = comp["gpu"].ns

                comp["gpu"].draw(pyglet.gl.GL_TRIANGLES)
            except:
                GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
                
                for attr in ["Kd", "Ks", "Ka"]: self.pipeline[attr] = comp["gpu"].__dict__[attr][:3]
                self.pipeline["shininess"] = comp["gpu"].ns

                comp["gpu"].draw(pyglet.gl.GL_TRIANGLES)
                
        pass