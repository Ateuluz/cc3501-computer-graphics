from elements.MyParticle import MyParticle

class MyParticleGenerator():
    
    def __init__(self, program_state, gen_particle_func, **particle_object_data):
        self.program_state = program_state
        self.particles = dict()
        self.generate_func = gen_particle_func
        self.p_data = particle_object_data
        self.data = dict()
        pass
    
    def add_particle(self, draw_func, data):
        ps = self.program_state
        t = ps["total_time"]
        particle = MyParticle(program_state=ps,
                              **self.p_data,
                              )
        particle.data = data
        particle.transform_func = draw_func
        self.particles[t] = particle
    
    def generate(self):
        self.generate_func(
            self
        )
    
    def draw(self, force=False, **data):
        for t, p in self.particles.items():
            p.draw(force, **data)
    
    def update(self):
        ps = self.program_state
        tt = ps["total_time"]
        for t, p in self.particles.copy().items():
            if t < tt-p.span:
                del self.particles[t]
        self.generate()