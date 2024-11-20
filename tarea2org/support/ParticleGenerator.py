import numpy as np

import grafica.transformations as tr

from Particle import Particle

class ParticleGenerator(object):
    
    def __init__(self, count, gen_pos_func, **kwargs):
        """ kwargs: (pipeline, mesh_path, transform, parent)
        (pipeline: ShaderProgram,
         mesh_path: Any,
         transform: NDArray[floating[_32Bit]] = tr.identity(),
         parent: Any | None = None) """
         
        self.count = count
        self.pos_func = gen_pos_func #TODO?
        # self.create()
        
        self.particle_data = kwargs
        
    def create(self, include_func):
        """ include_func assures the particles are added """
        for i in range(self.count):
            particle = Particle(self.particle_data)
            include_func(particle)
            #particle.transform_func = self.pos_func #TODO

def ex_gen_pos_func(particle,t,data):
    """ A function that moves a particle in a parabolic fall until h=0 """
    
    if not "start_pos" in particle.data.keys():
        particle.data["start_pos"] = data["start_pos"] if "start_pos" in data.keys() else np.zeros(3)
    if not "pos" in particle.data.keys():
        particle.data["pos"] = particle.data["start_pos"]
    if not "start_vel" in particle.data.keys():
        particle.data["start_vel"] = data["start_vel"] if "start_vel" in data.keys() else np.zeros(3)
    if not "start_t" in particle.data.keys():
        particle.data["start_t"] = t
    if not "vel" in particle.data.keys():
        particle.data["vel"] = particle.data["start_vel"]
    
    particle.data["vel"][0] = particle.data["start_vel"][0] * np.exp(t-particle.data["start_t"])
    particle.data["vel"][1] = particle.data["start_vel"][1] * np.exp(t-particle.data["start_t"])
    particle.data["vel"][2] -= data["gravity"] if "gravity" in data.keys() else 0
    
    particle.data["pos"] += particle.data["vel"]
    
    particle.data["pos"][2] = max(0, particle.data["pos"][2])
    
    return tr.identity() @ tr.translate(*particle.data["pos"])