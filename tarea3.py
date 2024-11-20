import os
import sys
from pathlib import Path

import numpy as np
import pyglet
import pyglet.gl as GL
import trimesh as tm

import pymunk

if sys.path[0] != "":
    sys.path.insert(0, "")
    
import grafica.transformations as tr

from tarea2org.support.Light        import Light
from tarea2org.support.Projection   import Projection
from tarea2org.support.View         import View

from elements.MyObject            import MyObject           
from elements.MyParticleGenerator import MyParticleGenerator

import random


if __name__ == '__main__':
# Set Window
#==============================================================================
    window_width  = 800
    window_height = 800
    window = pyglet.window.Window(window_width,window_height)
    
# Set Program State
#==============================================================================
    window.program_state = {
        "total_time": 0.0,
        "elements_display": [],
        "mesh_scale": 1,
        "view": View(),
        "projection": Projection(),
        "light_1": Light(np.array([ 4, 5.5, 4]), np.array([( 0, .19, .4),( 0, .19, .4),( 0, .19, .4)])), 
        "light_2": Light(np.array([-4, 5.3, 4]), np.array([(.4, .19,  0),(.4, .19,  0),(.4, .19,  0)])),
        "light_3": Light(np.array([10, 2, 10]),    np.array([(.2, .05, .0),(.2, .05, .0),(.2, .05, .0)])),
        "pipelines": [],
        "pipeline_id": 0,
        "zoom_target": 0,
        "zoom_dimmer": 0,
        "show_floor" : 1,
        "pressed"    : dict()
    }
    ps = window.program_state
    
# Set Projections
#==============================================================================
    def perspective(fovy, aspect, near, far):
        f = 1.0 / np.tan(np.radians(fovy) / 2)
        nf = 1 / (near - far)
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) * nf, -1],
            [0, 0, (2 * far * near) * nf, 0]
        ], dtype=np.float32)
        
    def perspective_projection_matrix(fov, aspect, near, far):
        """
        Creates a perspective projection matrix.
        
        Parameters:
        - fov: Field of view in the y direction, in degrees.
        - aspect: Aspect ratio (width / height).
        - near: Near clipping plane distance.
        - far: Far clipping plane distance.
        
        Returns:
        - A 4x4 perspective projection matrix.
        """
        f = 1 / np.tan(np.radians(fov) / 2)
        nf = 1 / (near - far)
        
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) * nf, (2 * far * near) * nf],
            [0, 0, -1, 0]
        ], dtype=np.float32)
    
    def my_perspective(fov, aspect, near, far):
        f = 1 / np.tan(np.radians(fov) / 2)
        nf = 1 / (near - far)
        
        # return np.array([
        #     [f / aspect, 0, 0, 0],
        #     [0, f, 0, 0],
        #     [0, 0, (far + near) * nf, (2 * far * near) * nf],
        #     [0, 0, -1, 0]
        # ], dtype=np.float32)
        
        # return np.array([
        #     [f / aspect, 0, 0, 0],
        #     [0, f, 0, 0],
        #     [0, 0, -(far + near) * nf, -(2 * far * near) * nf],
        #     [0, 0, -1, 0]
        # ], dtype=np.float32)
        
        tg = np.tan(np.radians(fov) / 2)
        r  = near * tg
        t  = r / aspect
        return np.array([
            [near/r, 0, 0, 0],
            [0, near/t, 0, 0],
            [0, 0, (far + near) * nf, (2 * far * near) * nf],
            [0, 0, -1, 0]
        ], dtype=np.float32)
    
    ps["projection"].add(
        "perspectiveTest2",
        perspective(65, float(window_width)/float(window_height), -100, 110)
    )
    ps["projection"].add(
        "perspectiveTest3",
        perspective_projection_matrix(65, float(window_width)/float(window_height), 1, 1000)
    )
    ps["projection"].add(
        "perspectiveTest4",
        my_perspective(60, float(window_width)/float(window_height), 1, 10000)
    )
    ps["projection"].add(
        "perspectiveTest5",
        tr.perspective(55, float(window_width)/float(window_height), 1, 1000)
    )

# Set Views
#==============================================================================
    # We can change look_to and look_from
    # ps["view"].add(
    #     0,
    #     np.array([1, 1, 1]),
    #     np.array([0.0, 0.0, 0.0]),
    #     "perspectiveTest2"
    # )
    
    ps["view"].add(
        1,
        np.array([2.5, 0, 0]),
        np.array([0.0, 0.0, 0.0]),
        "perspectiveTest5"
    )
    
    ps["view"].add(
        0,
        np.array([1, 1, 1]),
        np.array([0.0, 0.0, 0.0]),
        "perspectiveTest5"
    )

# Set Pipeline
#==============================================================================
    # Vertex and Fragment
    with open(
        Path(os.path.dirname(__file__))
        / "vertex_program.glsl"
    ) as f:
        vertex_source_code = f.read()

    with open(
        Path(os.path.dirname(__file__))
        / "fragment_program.glsl"
    ) as f:
        fragment_source_code = f.read()

    # Create Pipeline
    vert_shader = pyglet.graphics.shader.Shader(vertex_source_code  , "vertex")
    frag_shader = pyglet.graphics.shader.Shader(fragment_source_code, "fragment")
    flipper_pipeline = pyglet.graphics.shader.ShaderProgram(vert_shader, frag_shader)
    ps["pipelines"].append(flipper_pipeline)

# Set Elements
#==============================================================================
    # Get Reference
    ref = tm.load("my_assets/tri_stoneFloor.obj").scale
    ps["mesh_scale"] = ps["mesh_scale"] / ref
    scale = ps["mesh_scale"]
    
    # Floor
    floor = MyObject(
        program_state= ps,
        pipeline  = flipper_pipeline,
        mesh_path = "my_assets/tri_stoneFloor.obj",
        transform = tr.uniformScale(scale) @ tr.uniformScale(1.5),
        position  = np.zeros(3)
    )
    ps["elements_display"].append(floor)

    # Ball
    main_ball = MyObject(
        program_state= ps,
        pipeline  = flipper_pipeline,
        mesh_path = "my_assets/tri_metBall.obj",
        transform = tr.uniformScale(scale) @ tr.uniformScale(1/4),
        position  = np.array([0,.99+0.009,0])
    )
    main_ball.set_gpu(True,
                 np.array([700,700,700]),
                 np.array([999,999,999]),
                 np.array([100,100,100]),
                 32)
    ps["elements_display"].append(main_ball)
    
# Set Ball Particles
#==============================================================================
    def ground_particle_generator(self:MyParticleGenerator):
        if not 1 in ps["pressed"]: return
        margin = .15
        # if len(self.particles.keys()) == 0 or ps["total_time"] - max(self.particles.keys()) > margin:
        if len(self.particles.keys()) == 0 or ps["total_time"] - list(self.particles.keys())[-1] > margin:
            data = dict()
            
            data["pos"] = main_ball.Vec3pos + np.array([0,1,0]) * np.sin(ps["total_time"]*10) * .01
            
            self.add_particle(ground_particle_position, data)
        pass
    
    def ground_particle_position(self):
        d = self.data
        
        self.size = (.7 - (ps["total_time"] - self.born)) * .5
        self.Vec3pos = d["pos"]
        
        return tr.translate(*self.Vec3pos) @ tr.uniformScale(self.size)

    main_ball.particleGenerators[
        "ground"
    ] = MyParticleGenerator(
        gen_particle_func= ground_particle_generator,
        program_state= ps,
        pipeline  = flipper_pipeline,
        mesh_path = "my_assets/tri_metBall.obj",
        transform = tr.uniformScale(scale) @ tr.uniformScale(1/4),
        position  = np.zeros(3),
        span      = .6
    )
    
    def orbit_particle_generator(self):
        if not 2 in ps["pressed"]: return
        if not "count" in self.data.keys():
            self.data["count"] = 0
        n = 3
        c = len(self.particles.keys())
        if c < n:
            data = dict()
            
            i = self.data["count"]
            self.data["count"] += 1
            
            data["pos"] = main_ball.Vec3pos
            data["off"] = np.pi * 2 * n**-1 * i
            
            self.add_particle(orbit_particle_position, data)
            
            ps["total_time"] += .0001
    
    def orbit_particle_position(self):
        s = self.span - .0  # span = .2
        
        t = (ps["total_time"] - self.born)
        
        self.size = .05 + s * (self.span - t) / self.span / s * .2
        
        c = .03
        dist = np.sin(t/s * np.pi * 2) * c
        h = 0.04 + (t - s / 2) / s * c * 5
        a = self.data["off"] + ps["total_time"] * 20
        self.Vec3pos = main_ball.Vec3pos
        doff = np.array(
            [
                dist,
                h,
                0
            ]
        )
        
        return tr.translate(*self.Vec3pos) @ tr.rotationY(a) @ tr.translate(*doff) @ tr.uniformScale(self.size)
    
    main_ball.particleGenerators[
        "orbit"
    ] = MyParticleGenerator(
        gen_particle_func= orbit_particle_generator,
        program_state= ps,
        pipeline  = flipper_pipeline,
        mesh_path = "my_assets/tri_metBall.obj",
        transform = tr.uniformScale(scale) @ tr.uniformScale(1/4),
        position  = np.zeros(3),
        span      = .8
    )


# Set Key Actions
#==============================================================================
    key = pyglet.window.key
    
    @window.event
    def on_key_press(symbol,modifiers):
        view:   View = ps["view"]
        if symbol == key.C:
            view.next()
            ps["projection"].current = view.projection
        if symbol == key.V:
            ps["show_floor"] += 1
            ps["show_floor"] %= 2
        elif symbol == key.SPACE:
            ps["zoom_target"] = 1
        elif symbol == key.NUM_1:
            if 1 in ps["pressed"]:
                del ps["pressed"][1]
            else:
                ps["pressed"][1] = True
        elif symbol == key.NUM_2:
            ps["pressed"][2] = True
        pass
    
    @window.event
    def on_key_release(symbol,modifiers):
        view:   View = ps["view"]
        if symbol == key.SPACE:
            ps["zoom_target"] = 0
            # restart()
            pass
        elif symbol == key.NUM_2:
            if 2 in ps["pressed"]:
                del ps["pressed"][2]
        pass
    pass

# Set Auxiliary Functions
#==============================================================================
    def light_3_position():
        t = ps["total_time"] * 30
        x,y,z = ps["light_3"].pos
        return np.array([z*np.sin(t), y, z*np.cos(t)])
    
    def mult_color_3():
        # t0 = ps["total_time"]
        # t1 = ps["last_collision"]
        # return (abs(t0 - t1) < 2)
        return True # t0 // 2 % 2 == 0
    
    def get_loop_demo_position(offset):
        t = ps["total_time"] + offset
        return np.array(
            [
                np.sin(t * 3.0) * .2,
                0.03,
                np.cos(t * 1.5) * .2 * 2
            ]
        )
    
    def support_view():
        view: View = ps["view"]
        st = ps["zoom_target"]
        s  = ps["zoom_dimmer"]
        c  = .0003 + (1 - st) / 20
        
        ps["zoom_dimmer"] += (st - s) * (c + s / 10)
        s  = ps["zoom_dimmer"]
        
        target = main_ball.Vec3pos.copy()
        mul_c = 1
        target[0] *= mul_c
        target[2] *= mul_c
        target[1]  = 0
        to =   target    * s + view.look_to   * (1 - s)
        fr = (get_loop_demo_position(-.5) + np.array([0,1.2,0])) * s + view.look_from * (1 - s)
        
        return tr.lookAt(
            np.array(fr),
            np.array(to),
            np.array([0.0, 1.0, 0.0])
        )
        
    def support_view_vec3():
        view: View = ps["view"]
        s  = ps["zoom_dimmer"]
        
        target = main_ball.Vec3pos.copy()
        mul_c = 1
        target[0] *= mul_c
        target[2] *= mul_c
        target[1]  = 0
        fr = (get_loop_demo_position(-.5) + np.array([0,1.2,0])) * s + view.look_from * (1 - s)
        
        return fr
    
    def ball_t_func(self):
        # t_get = lambda: ps["total_time"]
        # t = t_get()
        rt = self.root.copy()
        rt @= tr.translate(*self.Vec3pos)
        return rt
        

# Set Draw
#==============================================================================
    @window.event
    def on_draw():
        GL.glClearColor(0.0, 0.6, 0.5, 1.0)
        GL.glLineWidth(1.0)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_DEPTH_TEST)
        window.clear()
        
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        pipeline = ps["pipelines"][ps["pipeline_id"]]
        pipeline.use()
        
        pipeline["lightPosition1"] = ps["light_1"].pos
        pipeline["lightPosition2"] = ps["light_2"].pos
        pipeline["lightPosition3"] = light_3_position()
        
        
        for i,x in zip(range(3),("a","s","d")):
            pipeline["L"+x+"1"] = ps["light_1"].color[i]
            pipeline["L"+x+"2"] = ps["light_2"].color[i]
            pipeline["L"+x+"3"] = ps["light_3"].color[i] * mult_color_3()
        
        projection:Projection = ps["projection"].get()
        
        pipeline["projection"] = projection
        pipeline["viewPosition"] = support_view_vec3()
        pipeline["view"] = support_view().reshape(16, 1, order="F")
        
        elements:list[MyObject] = ps["elements_display"][ps["show_floor"]:]
        for element in elements: element.draw(
            # force    = shine_gpu_conditions(element),
            # Ka       = shine_gpu_elements["Ka"],
            # Ks       = shine_gpu_elements["Ks"],
            # Kd       = shine_gpu_elements["Kd"],
            # shininess= shine_gpu_elements["shininess"],
        )
        
        shine_gpu_elements = {
            "Ka":np.array([1.9,1.8,.3]) * 2.0, # Lo intenté
            "Ks":np.array([1.9,1.8,.3]) * 3.0,
            "Kd":np.array([1.9,1.8,.3]) * 0.5,
            "shininess":8,
        }
        for tp, p_generator in main_ball.particleGenerators.items():
            p_generator.draw(
                force    = True,
                Ka       = shine_gpu_elements["Ka"],
                Ks       = shine_gpu_elements["Ks"],
                Kd       = shine_gpu_elements["Kd"],
                shininess= shine_gpu_elements["shininess"],
            )
        
        pass
    
# Set Update World
#==============================================================================
    def update_world(dt, window):
        window.program_state["total_time"] += dt
        
        # n = 10
        # for _ in range(n):
        #     pass
        main_ball.Vec3pos = get_loop_demo_position(0)
        
        for tp, p_generator in main_ball.particleGenerators.items():
            p_generator.update()
        
        pass
    
# Begin
#==============================================================================
    print(">>",main_ball.root)
    print(">>",main_ball.__dict__)
    
    main_ball.transform_func = ball_t_func
    
    ps["projection"].current = ps["view"].projection
    pyglet.clock.schedule_interval(update_world, 1 / 60.0, window)
    pyglet.app.run(1 / 60.0)
    
    

    pass