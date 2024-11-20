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

from grafica.textures import texture_2D_setup

# My stuff
from tarea2org.support.Element      import Element
from tarea2org.support.Projection   import Projection
from tarea2org.support.View         import View
from tarea2org.support.Light        import Light
from tarea2org.support.SceneElement import SceneElement

import pprint
import random

want_to_print = False
def my_print(*args, **kwargs):
    if want_to_print: print(*args, **kwargs)

if __name__ == "__main__":
    
# Set Window
    window_width  = 800
    window_height = 800
    window = pyglet.window.Window(window_width,window_height)

# Set World
    world = pymunk.Space()
    world.gravity = 0,0
    world.gravity = (0.0, .5)

# Set Program State
    window.program_state = {
        "total_time": 0.0,
        "elements_display": [],
        "elements_interact": [],
        "view": View(),
        "projection": Projection(),
        "mesh_scale": 1,
        "test_scale": 1/2,
        "light_1": Light(np.array([ 4, 5.5, 4]), np.array([( 0, .19, .4),( 0, .19, .4),( 0, .19, .4)])), 
        "light_2": Light(np.array([-4, 5.3, 4]), np.array([(.4, .19,  0),(.4, .19,  0),(.4, .19,  0)])),
        "light_3": Light(np.array([10, 2, 10]),    np.array([(.2, .05, .0),(.2, .05, .0),(.2, .05, .0)])),
        "last_collision": -999,
        "score": 0,
        "score_add": 1,
        "pipelines": [],
        "pipeline_id": 0,
        "game_on": True
    }
    ps = window.program_state

# Set Projections
    def perspective(fovy, aspect, near, far):
        f = 1.0 / np.tan(np.radians(fovy) / 2)
        nf = 1 / (near - far)
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) * nf, -1],
            [0, 0, (2 * far * near) * nf, 0]
        ], dtype=np.float32)
    
    def fisheye_projection(fov, aspect_ratio, near_clip, far_clip):
        projection_matrix = np.zeros((4, 4), dtype=np.float32)
        
        # Assuming equidistant fisheye projection model
        fov_rad = np.radians(fov)
        focal_length = 1.0  # Adjust based on the desired effect
        
        projection_matrix[0, 0] = focal_length * np.cos(fov_rad / 2) / aspect_ratio
        projection_matrix[1, 1] = focal_length * np.cos(fov_rad / 2)
        projection_matrix[2, 2] = -(far_clip + near_clip) / (far_clip - near_clip)
        projection_matrix[2, 3] = -(2 * far_clip * near_clip) / (far_clip - near_clip)
        projection_matrix[3, 2] = -1.0
        
        return projection_matrix

    ps["projection"].add(
        "perspective",
        tr.perspective(60, float(window_width)/float(window_height), .01, 200)
    )
    ps["projection"].add(
        "perspective2",
        tr.perspective(90, float(window_width)/float(window_height), .01, 400)
    )
    ps["projection"].add(
        "perspectiveTest",
        perspective(60, float(window_width)/float(window_height), .01, 400)
    )
    ps["projection"].add(
        "perspectiveTest2",
        perspective(65, float(window_width)/float(window_height), -100, 110)
    )
    ps["projection"].add(
        "ortho",
        tr.ortho(-.6, .6, -.6, .6, .01, 10)
    )
    ps["projection"].add(
        "ortho2",
        tr.ortho(-.6, .6, -.6, .6, .01, 10)
    )
    ps["projection"].add(
        "orthoTest",
        tr.ortho(-.6, .6, -.6, .6, .01, 10)
    )
    ps["projection"].add(
        "fisheyeTest",
        fisheye_projection(100, float(window_width)/float(window_height), -100, 400),
    )

# Set Views
    ps["view"].add(
        5,
        np.array([0.0, 0.8, 0.01]) * .8,
        np.array([0.0, 0.0, -.02]) * .8,
        "perspectiveTest"
    )
    ps["view"].add(
        1,
        np.array([1.6, 1.6, 1.6]),
        np.array([0.0, 0.0, 0.0]),
        "ortho"
    )
    # ps["view"].add(
    #     0,
    #     np.array([0.0, 0.8, 0.01]) * .8,
    #     np.array([0.0, 0.0, -.02]) * .8,
    #     "perspective"
    # )
    # ps["view"].add(
    #     4,
    #     np.array([0.0, 0.8, 0.01]) * .8,
    #     np.array([0.0, 0.0, -.02]) * .8,
    #     "ortho2"
    # )
    ps["view"].add(
        2,
        np.array([0.0, 0.8, 0.5]),
        np.array([0.0, 0.0, 0.0]),
        "ortho"
    )
    # ps["view"].add(
    #     3,
    #     np.array([1.6, 1.6, 1.6]) * .1,
    #     np.array([0.0, 0.0, 0.0]),
    #     "perspective2"
    # )
    ps["view"].add(
        6,
        np.array([1, 1, 1]),
        np.array([0.0, 0.0, 0.0]),
        "perspectiveTest2"
    )
    # ps["view"].add(
    #     7,
    #     np.array([1, 1, 1]),
    #     np.array([0.0, 0.0, 0.0]),
    #     "orthoTest"
    # )
    # ps["view"].add(
    #     8,
    #     np.array([1, 1, 1]),
    #     np.array([0.0, 0.0, 0.0]),
    #     "fisheyeTest"
    # )
    
# Set Pipeline
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
    
    # Fisheye Vertex and Fragment
    with open(
        Path(os.path.dirname(__file__))
        / "tarea2org/fisheye_v_p.glsl"
    ) as f:
        vertex_source_code = f.read()

    with open(
        Path(os.path.dirname(__file__))
        / "tarea2org/fisheye_f_p.glsl"
    ) as f:
        fragment_source_code = f.read()

    # Create Pipeline
    vert_shader      = pyglet.graphics.shader.Shader(vertex_source_code  , "vertex")
    frag_shader      = pyglet.graphics.shader.Shader(fragment_source_code, "fragment")
    fisheye_pipeline = pyglet.graphics.shader.ShaderProgram(vert_shader, frag_shader)
    ps["pipelines"].append(fisheye_pipeline)
    
# Set Elements
    # Get Reference
    ref = tm.load("my_assets/tri_stoneFloor.obj").scale
    ps["mesh_scale"] = 2/ref * ps["test_scale"]
    scale = ps["mesh_scale"]
    
    # Floor
    floor = Element(
        pipeline  = flipper_pipeline,
        mesh_path = "my_assets/tri_stoneFloor.obj",
        transform = tr.uniformScale(scale)
    )
    ps["elements_display"].append(floor)
    
    # Walls
    walls = Element(
        pipeline  = flipper_pipeline,
        mesh_path = "my_assets/tri_fireWalls.obj",
        transform = tr.uniformScale(scale) @ tr.scale(1,1,.999),
        parent    = floor
    )
    ps["elements_display"].append(walls)

    # Flipper Factory
    # flipper_base_verts = np.array([(-0.016,.008),(-0.008,-0.016),(.054,-.016),(.054,-.004),(-0.016,.016)]) * 1.
    flipper_base_verts = np.array([(-0.016,-0.016),(.054,-.016),(.054,-.004),(-0.016,.016)]) * 1.
    flipper_base_verts = [pymunk.Vec2d(x,y) for x,y in flipper_base_verts]
    def new_flipper(vertices, transformation=None, collision_type= 0):
        if transformation is None: transformation = tr.identity() @ tr.uniformScale(scale)
        flipper = Element(
            pipeline  = flipper_pipeline,
            mesh_path = "my_assets/tri_orangeFlipper.obj",
            transform = transformation,
            parent    = floor
        )
        flipper.body  = pymunk.Body(body_type= pymunk.Body.KINEMATIC)
        flipper.shape = pymunk.Poly(flipper.body, vertices= vertices) # TODO
        flipper.body.angle    = 0
        flipper.body.position = (0,0)
        flipper.shape.elasticity = .05
        flipper.shape.collision_type = collision_type
        ps["elements_display" ].append(flipper)
        ps["elements_interact"].append(flipper)
        return flipper
    
    # Fizq
    flipper_izq = new_flipper(flipper_base_verts,
                              collision_type= 1)
    # Fder
    flipper_der = new_flipper([(-x,y) for x,y in flipper_base_verts],
                              transformation = tr.uniformScale(scale) @ tr.scale(-1,1,1),
                              collision_type= 2)
    # Fup
    flipper_up = new_flipper([(x*.65,y*.65) for x,y in flipper_base_verts],
                              collision_type= 3)
    
    # Mushroom Factory
    def new_mushroom(radius= .036, bounciness= 1, transformation=None, collision_type= 0):
        mushroom = Element(
            pipeline  = flipper_pipeline,
            mesh_path = "my_assets/tri_jpMushroom.obj",
            transform = tr.uniformScale(scale),
            parent    = floor
        )
        mushroom.body  = pymunk.Body(body_type= pymunk.Body.KINEMATIC)
        mushroom.shape = pymunk.Circle(mushroom.body, radius)
        mushroom.shape.elasticity = bounciness
        mushroom.shape.collision_type = collision_type
        ps["elements_display" ].append(mushroom)
        ps["elements_interact"].append(mushroom)
        return mushroom
        pass

    
    # Mizq
    mushroom_izq = new_mushroom(bounciness= 1.5,collision_type= 4)
    mushroom_der = new_mushroom(bounciness= 1.5,collision_type= 4)
    mushroom_dwn = new_mushroom(bounciness= 1.5,collision_type= 4)
    _mf = lambda n: n%5 == 0
    mushroom_support = [new_mushroom(.018, .8 + .4*_mf(i), collision_type=6*_mf(i)) for i in range(10)]
    
    ball = Element(
        pipeline  = flipper_pipeline,
        mesh_path = "my_assets/tri_metBall.obj",
        transform = tr.uniformScale(scale) @ tr.uniformScale(1/8),
        parent    = floor
    )
    ball.set_gpu(True,
                 np.array([400,400,400]), # Lo intenté
                 np.array([900,900,900]),
                 np.array([400,400,400]),
                 32)
    ball_mass  = 0.1
    ball_rad   = 0.014
    ball.body  = pymunk.Body(
        ball_mass,
        pymunk.moment_for_circle(ball_mass, 0, ball_rad)
    )
    ball.shape = pymunk.Circle(ball.body,ball_rad)
    ball.shape.elasticity = .8
    ball.shape.collision_type = 0
    ps["elements_display" ].append(ball)
    ps["elements_interact"].append(ball)

# Set F1 Try #BUG
    f1_1 = SceneElement(
        pipeline= flipper_pipeline,
        mesh_path="my_assets/41-formula-1/formula 1/Formula 1 mesh.obj",
        transform= tr.uniformScale(scale)
    )
    ps["elements_display"].append(f1_1)

# Set World Elements
    for element in ps["elements_interact"]:
        world.add(element.body, element.shape)
    # world.add(*ps["elements_interact"])
    
# Set Borders
    def add_static_to_wrld(a,b, elasticity = .7, group = 1):
        line = pymunk.Segment(world.static_body, a, b, .0001)
        line.elasticity = elasticity
        line.group = group
        world.add(line)
        return line
    
    # TODO: Add borders
    bottom_line = add_static_to_wrld((-0.248, 0.418),( 0.248, 0.418))
    add_static_to_wrld(( 0.248, 0.418),( 0.248,-0.418))
    add_static_to_wrld(( 0.248,-0.418),(-0.248,-0.418))
    add_static_to_wrld((-0.248,-0.418),(-0.248, 0.418))
    bottom_line.collision_type = 5
    
# Set Roots
    floor.transform_root(
        # ry=(np.pi)
         s = 1.2
    )
    ball.root_set()
    ball.transform_root(
        ty= .01,
        tz= 0,
    )
    ball.move_body((-.3+.05) *ps["test_scale"],(-.5-.05) *ps["test_scale"])
    walls.root_set()
    walls.transform_root(
        ty= -.001,
        s= .999
    )
    flipper_izq.root_set()
    flipper_izq.transform_root(
        tx= -.15*ps["test_scale"],
        tz=  .61*ps["test_scale"],
        s =  .8
    )
    flipper_der.root_set()
    flipper_der.transform_root(
        tx=  .15*ps["test_scale"],
        tz=  .61*ps["test_scale"],
        s =  .8
    )
    flipper_up.root_set()
    flipper_up.transform_root(
        tx= -.3 *ps["test_scale"],
        tz= -.5 *ps["test_scale"],
        s =  .5
    )
    mushroom_izq.root_set()
    mushroom_izq.transform_root(
        tx= -.2*ps["test_scale"],
        tz= -.2*ps["test_scale"]
    )
    mushroom_der.root_set()
    mushroom_der.transform_root(
        tx=  .2*ps["test_scale"],
        tz= -.2*ps["test_scale"]
    )
    mushroom_dwn.root_set()
    
    support_coors = [(-.39,.34),(-.36,.41),(-.322,.472),(-.277,.53),(-.22,.57),(.39,.34),(.36,.41),(.322,.472),(.277,.53),(.22,.57),]
    for i,mush in enumerate(mushroom_support):
        x,z = support_coors[i]
        mush.root_set()
        mush.transform_root(
            tx=  x*ps["test_scale"],
            tz=  z*ps["test_scale"],
            s =  .5 #+ (.07 if i%3 == 1 else 0)
        )

# Set Auxiliary Functions
    def light_3_position():
        t = ps["total_time"] * 30
        x,y,z = ps["light_3"].pos
        return np.array([z*np.sin(t), y, z*np.cos(t)])
    
    def mult_color_3():
        t0 = ps["total_time"]
        t1 = ps["last_collision"]
        return (abs(t0 - t1) < 2)
        
# Set Special Functions
    def restart():
        if not ps["game_on"]:
            ball.move_body((-.3+0.05) *ps["test_scale"],(-.5-.06) *ps["test_scale"])
            ball.body.velocity = (0,-.2)
            window.program_state["total_time"] = 0
            print("Game Restart! Good Luck!")
            ps["last_collision"] = -999
            ps["total_time"]     = 0
            ps["score"]          = 0 
            ps["score_add"]      = 1
        else:
            print("Game is already running! Wait for the ball to fall!")
            print(f"Current score: {ps["score"]} points!")
        
    def close_call(*args,**kwargs):
        print("Close Call!")
        return True
        
    def game_over(arbiter,space,data):
        print(f"{'Game Over! Thanks For Playing!'}")
        print(f"{'Play Tyme:'} {ps["total_time"]:.03} seconds!")
        print(f"{'Score:'} {ps["score"]} points!")
        ball.body.velocity = 0,0
        ps["game_on"] = False
        return True
    
    def hit_main_mushroom(arbiter,space,data):
        print(f"Collision! {ps['score_add']} points added!")
        ps["last_collision"] = ps["total_time"]
        ps["score"]     += ps["score_add"]
        ps["score_add"] += 1
        return True
    
    def score_add_reset(*args,**kwargs):
        ps["score_add"] = 1
        return True
    
    def add_lame_collisions(*args):
        for i in args:
            add_collition(0,i,score_add_reset)
    
    def pipeline_next():
        ps["pipeline_id"] = (ps["pipeline_id"] + 1) % len(ps["pipelines"])

# Set Collitions
    def add_collition(a,b,f):
        c = world.add_collision_handler(a,b)
        c.begin = f
    
    add_lame_collisions(0,1,2,3,5,6)
    
    add_collition(0,6,close_call)
    add_collition(0,5,game_over)
    add_collition(0,4,hit_main_mushroom)
    
# Set Key Actions
    key = pyglet.window.key
    ang_vel = 22

    @window.event
    def on_key_press(symbol,modifiers):
        view   :View = ps["view"]
        if symbol == key.A:
            flipper_izq.body.angular_velocity = -ang_vel
            flipper_up.body.angular_velocity  = -ang_vel
        elif symbol == key.D:
            flipper_der.body.angular_velocity = ang_vel
        elif symbol == key.C:
            view.next()
            ps["projection"].current = view.projection
        elif symbol == key.P:
            # ball.move_body(0,0)
            # ball.body.velocity = 0,0
            pipeline_next()
        elif symbol == key.SPACE:
            # ball.move_body(-0.01,-0.2)
            # ball.move_body(-0.05,-0.3)
            restart()
        # elif symbol == key.UP:
        #     ball.body.position += pymunk.Vec2d(0,-.01) 
        #     print(ball.body.position)
        # elif symbol == key.DOWN:
        #     ball.body.position += pymunk.Vec2d(0,.01) 
        #     print(ball.body.position)
        # elif symbol == key.RIGHT:
        #     ball.body.position += pymunk.Vec2d(.01,0) 
        #     print(ball.body.position)
        # elif symbol == key.LEFT:
        #     ball.body.position += pymunk.Vec2d(-.01,0) 
        #     print(ball.body.position)
        # elif symbol == key.G:
        #     print(">> -=- <<")
        
    @window.event
    def on_key_release(symbol,modifiers):
        if symbol == key.A:
            flipper_izq.body.angular_velocity = ang_vel
            flipper_up.body.angular_velocity  = ang_vel
        elif symbol == key.D:
            flipper_der.body.angular_velocity = -ang_vel
            
# Set Special Draw Functions
    def ball_func(ball:Element):
        updated_root = ball.root.copy()
        x,z = ball.body.position
        updated_root @= tr.translate(x, 0, z)
        return updated_root
    
    def flipper_func(flipper:Element): # TODO
        updated_root = flipper.root.copy() @ tr.identity()
        angle = flipper.body.angle
        updated_root @= tr.rotationY(-angle)
        return updated_root

# Set Draw
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
        
        pipeline["La1"] = ps["light_1"].color[0]
        pipeline["La2"] = ps["light_2"].color[0]
        pipeline["La3"] = ps["light_3"].color[0] * mult_color_3()
        pipeline["Ls1"] = ps["light_1"].color[1]
        pipeline["Ls2"] = ps["light_2"].color[1]
        pipeline["Ls3"] = ps["light_3"].color[1] * mult_color_3()
        pipeline["Ld1"] = ps["light_1"].color[2]
        pipeline["Ld2"] = ps["light_2"].color[2]
        pipeline["Ld3"] = ps["light_3"].color[2] * mult_color_3()
        
        view:View             = ps["view"]
        projection:Projection = ps["projection"].get()
        
        pipeline["projection"] = projection
        pipeline["viewPosition"] = view.vec3from()
        pipeline["view"] = view.matrix().reshape(16, 1, order="F")
        
        ball        .transform_func = ball_func
        flipper_izq .transform_func = flipper_func
        flipper_der .transform_func = flipper_func
        flipper_up  .transform_func = flipper_func
        
        shine_gpu_elements = {
            "Ka":np.array([.7,.9,.8]), # Lo intenté
            "Ks":np.array([.7,.9,.8]),
            "Kd":np.array([.7,.9,.8]),
            "shininess":1,
            }
        
        shine_gpu_conditions = lambda e: (
            mult_color_3() and
            (e in mushroom_support) and 
            (mushroom_support.index(e) % 5 == int(
                (ps["total_time"] - ps["last_collision"]) * 10
            ) % 5)
        )
        
        elements:list[Element] = ps["elements_display"]
        for element in elements: element.draw(
            force    = shine_gpu_conditions(element),
            Ka       = shine_gpu_elements["Ka"],
            Ks       = shine_gpu_elements["Ks"],
            Kd       = shine_gpu_elements["Kd"],
            shininess= shine_gpu_elements["shininess"],
        )
        
# Set Update World
    def update_world(dt, window):
        window.program_state["total_time"] += dt
        
        margin = 0.0001
        
        n = 10
        for _ in range(n):
            world.step(dt / n)
        
        flipper_izq.body.angle = max(-np.pi * 3 / 16, min(np.pi * 3 / 16, flipper_izq.body.angle))
        if abs(flipper_izq.body.angle) + margin > np.pi * 3 / 16:
            flipper_izq.body.angular_velocity = 0
        flipper_der.body.angle = max(-np.pi * 3 / 16, min(np.pi * 3 / 16, flipper_der.body.angle))
        if abs(flipper_der.body.angle) + margin > np.pi * 3 / 16:
            flipper_der.body.angular_velocity = 0
        flipper_up.body.angle = max(-0., min(np.pi / 2, flipper_up.body.angle))
        if (flipper_up.body.angle - margin < 0) or (flipper_up.body.angle + margin > (np.pi / 2)):
            flipper_up.body.angular_velocity = 0

                
# Begin
    flipper_izq.body.angular_velocity =  ang_vel
    flipper_up .body.angular_velocity =  ang_vel
    flipper_up .body.angle            =  0.001
    flipper_der.body.angular_velocity = -ang_vel
    ball.body.velocity = 0,-.3
    ps["projection"].current = ps["view"].projection
    pyglet.clock.schedule_interval(update_world, 1 / 60.0, window)
    pyglet.app.run(1 / 60.0)