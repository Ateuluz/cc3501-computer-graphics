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

import pprint

if __name__ == "__main__":
    window_width  = 800
    window_height = 800
    window = pyglet.window.Window(
        window_width,
        window_height
    )
    
    
    # Necesitamos elementos esn la escena
    flipper_floor = tm.load("my_assets/tri_stoneFloor.obj")
    flipper_walls_tall = tm.load("my_assets/tri_fireWalls.obj")
    flipper = tm.load("my_assets/tri_orangeFlipper.obj")
    flipper_mushroom = tm.load("my_assets/tri_jpMushroom.obj")
    flipper_ball = tm.load("my_assets/tri_metBall.obj") # TODO Check
    # f1car = tm.load("my_assets/tri_f1car.obj")
    
    
    # Los escalamos
    scale_mult = 1/2
    scale = 2/flipper_floor.scale*scale_mult
    
    
    # generamos las transformaciones
    flipper_floor_scale = tr.uniformScale(scale)
    flipper_floor_rotate = tr.rotationX(0)#np.pi/2)
    flipper_floor.apply_transform(flipper_floor_rotate @ flipper_floor_scale)
    
    flipper_walls_tall_scale = tr.uniformScale(scale)
    flipper_walls_tall_rotate = tr.rotationX(0)#np.pi/2)
    flipper_walls_tall.apply_transform(flipper_walls_tall_rotate @ flipper_walls_tall_scale)
    
    flipper_mushroom_scale = tr.uniformScale(scale)
    flipper_mushroom_rotate = tr.rotationX(0)#np.pi/2)
    flipper_mushroom.apply_transform(flipper_mushroom_rotate @ flipper_mushroom_scale)
    
    flipper_scale = tr.uniformScale(scale)
    flipper_rotate = tr.rotationX(0)#np.pi/2)
    flipper.apply_transform(flipper_rotate @ flipper_scale)
    
    flipper_ball_scale = tr.uniformScale(scale)
    flipper_ball.apply_transform(flipper_ball_scale @ tr.uniformScale(1/8))
    
    # TODO
    # f1car_scale = tr.uniformScale(scale)
    # f1car_rotate = tr.rotationX(-np.pi/2)
    # f1car.apply_transform(f1car_rotate @ f1car_scale)
    
    
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
    vert_shader = pyglet.graphics.shader.Shader(vertex_source_code, "vertex")
    frag_shader = pyglet.graphics.shader.Shader(fragment_source_code, "fragment")
    flipper_pipeline = pyglet.graphics.shader.ShaderProgram(vert_shader, frag_shader)
    
    
    # ===============================================================================
    # ===============================================================================
    
    flipper_vertex_list = tm.rendering.mesh_to_vertexlist(flipper)
    num_vertices = len(flipper_vertex_list[4][1]) // 3
    flipper_gpu = flipper_pipeline.vertex_list_indexed(
        num_vertices, GL.GL_TRIANGLES, flipper_vertex_list[3],
        # position=flipper_vertex_list[4][1],
        # normal=flipper_vertex_list[5][1],
        # uv=flipper_vertex_list[6][1] 
    )
    flipper_gpu.position[:] = flipper_vertex_list[4][1]
    
    # # TODO
    # # Debug prints for buffer sizes
    # print("Number of vertices (positions):", len(flipper_gpu.position) // 3)
    # print("Number of normals:", len(flipper_gpu.normal) // 3)
    # print("Number of UVs (expected):", len(flipper_gpu.uv) // 2)
    # expected_uv_size = num_vertices * 2
    # if len(flipper_gpu.uv) != expected_uv_size:
    #     print(f"Reinitializing UV buffer to expected size: {expected_uv_size}")
    #     flipper_gpu.uv = pyglet.graphics.vertexbuffer.create_buffer(expected_uv_size)
    
    # #TODO Debug
    # print("UV size:", len(flipper_vertex_list[6][1]))
    # print("Expected UV size:", len(flipper_gpu.uv))
    
    # # Verify UV data TODO
    # expected_uv_count = len(flipper_vertex_list[4][1]) // 3
    # actual_uv_count = len(flipper_vertex_list[6][1]) // 2

    # # Debug prints for counts TODO
    # print("Number of vertices:", expected_uv_count)
    # print("Number of UVs:", actual_uv_count)
    
    flipper_gpu.normal[:] = flipper_vertex_list[5][1]
    flipper_gpu.uv[:] = flipper_vertex_list[6][1]
    flipper_gpu.texture = texture_2D_setup(flipper.visual.material.image)
    
    flipper_gpu.Ka = flipper.visual.material.__dict__["ambient"]
    flipper_gpu.Kd = flipper.visual.material.__dict__["diffuse"]
    flipper_gpu.Ks = flipper.visual.material.__dict__["specular"]
    flipper_gpu.ns = flipper.visual.material.glossiness
    #flipper_gpu.ns = flipper.visual.material.__dict__["glossiness"]
    
    # TO DO: Understand this
    # ===============================================================================
    
    # for attr in flipper.visual.material.__dict__:
    #     #print(f"{attr} = {flipper.visual.material.__dict__[attr]}")
    #     match attr:
    #         case "ambient":
    #             #print(f"{attr} = ambient")
    #             flipper_gpu.Ka = flipper.visual.material.__dict__[attr]
    #         case "diffuse":
    #             #print(f"{attr} = diffuse")
    #             flipper_gpu.Kd = flipper.visual.material.__dict__[attr]
    #         case "specular":
    #             #print(f"{attr} = specular")
    #             flipper_gpu.Ks = flipper.visual.material.__dict__[attr]
    #         case "kwargs":
    #             flipper_gpu.ns = flipper.visual.material.__dict__[attr]["ns"]
    # ===============================================================================
    
    
    flipper_floor_vertex_list = tm.rendering.mesh_to_vertexlist(flipper_floor)
    flipper_floor_gpu = flipper_pipeline.vertex_list_indexed(
        len(flipper_floor_vertex_list[4][1]) // 3, GL.GL_TRIANGLES, flipper_floor_vertex_list[3]
    )
    flipper_floor_gpu.position[:] = flipper_floor_vertex_list[4][1]
    
    flipper_floor_gpu.normal[:] = flipper_floor_vertex_list[5][1]
    flipper_floor_gpu.uv[:] = flipper_floor_vertex_list[6][1]
    flipper_floor_gpu.texture = texture_2D_setup(flipper_floor.visual.material.image)
    
    flipper_floor_gpu.Ka = flipper_floor.visual.material.__dict__["ambient"]
    flipper_floor_gpu.Kd = flipper_floor.visual.material.__dict__["diffuse"]
    flipper_floor_gpu.Ks = flipper_floor.visual.material.__dict__["specular"]
    flipper_floor_gpu.ns = flipper_floor.visual.material.glossiness
    
    
    flipper_walls_tall_vertex_list = tm.rendering.mesh_to_vertexlist(flipper_walls_tall)
    flipper_walls_tall_gpu = flipper_pipeline.vertex_list_indexed(
        len(flipper_walls_tall_vertex_list[4][1]) // 3, GL.GL_TRIANGLES, flipper_walls_tall_vertex_list[3]
    )
    flipper_walls_tall_gpu.position[:] = flipper_walls_tall_vertex_list[4][1]
    
    flipper_walls_tall_gpu.normal[:] = flipper_walls_tall_vertex_list[5][1]
    flipper_walls_tall_gpu.uv[:] = flipper_walls_tall_vertex_list[6][1]
    flipper_walls_tall_gpu.texture = texture_2D_setup(flipper_walls_tall.visual.material.image)
    
    flipper_walls_tall_gpu.Ka = flipper_walls_tall.visual.material.__dict__["ambient"]
    flipper_walls_tall_gpu.Kd = flipper_walls_tall.visual.material.__dict__["diffuse"]
    flipper_walls_tall_gpu.Ks = flipper_walls_tall.visual.material.__dict__["specular"]
    flipper_walls_tall_gpu.ns = flipper_walls_tall.visual.material.glossiness
    
    
    flipper_mushroom_vertex_list = tm.rendering.mesh_to_vertexlist(flipper_mushroom)
    flipper_mushroom_gpu = flipper_pipeline.vertex_list_indexed(
        len(flipper_mushroom_vertex_list[4][1]) // 3, GL.GL_TRIANGLES, flipper_mushroom_vertex_list[3]
    )
    flipper_mushroom_gpu.position[:] = flipper_mushroom_vertex_list[4][1]
    
    flipper_mushroom_gpu.normal[:] = flipper_mushroom_vertex_list[5][1]
    flipper_mushroom_gpu.uv[:] = flipper_mushroom_vertex_list[6][1]
    flipper_mushroom_gpu.texture = texture_2D_setup(flipper_mushroom.visual.material.image)
    
    flipper_mushroom_gpu.Ka = flipper_mushroom.visual.material.__dict__["ambient"]
    flipper_mushroom_gpu.Kd = flipper_mushroom.visual.material.__dict__["diffuse"]
    flipper_mushroom_gpu.Ks = flipper_mushroom.visual.material.__dict__["specular"]
    flipper_mushroom_gpu.ns = flipper_mushroom.visual.material.glossiness
    
    
    
    flipper_ball_vertex_list = tm.rendering.mesh_to_vertexlist(flipper_ball)
    flipper_ball_gpu = flipper_pipeline.vertex_list_indexed(
        len(flipper_ball_vertex_list[4][1]) // 3, GL.GL_TRIANGLES, flipper_ball_vertex_list[3]
    )
    flipper_ball_gpu.position[:] = flipper_ball_vertex_list[4][1]
    
    flipper_ball_gpu.normal[:] = flipper_ball_vertex_list[5][1]
    
    print("Exp",len(flipper_ball_gpu.uv[:]))
    print("Act",len(flipper_ball_vertex_list[6][1]))
    flipper_ball_gpu.uv[:] = flipper_ball_vertex_list[6][1][:3968]
    flipper_ball_gpu.texture = texture_2D_setup(flipper_ball.visual.material.image)
    
    flipper_ball_gpu.Ka = flipper_ball.visual.material.__dict__["ambient"]
    flipper_ball_gpu.Kd = flipper_ball.visual.material.__dict__["diffuse"]
    flipper_ball_gpu.Ks = flipper_ball.visual.material.__dict__["specular"]
    flipper_ball_gpu.ns = flipper_ball.visual.material.glossiness
    
    print(flipper_ball_gpu.Ks)
    
    flipper_ball_gpu.Ks = 255 - (255 - flipper_ball_gpu.Ks)/4
    flipper_ball_gpu.Ka = 255 - (255 - flipper_ball_gpu.Ka)/4
    print(flipper_ball_gpu.Ks)
    
    
    # f1car_vertex_list_dict = dict()
    # f1car_gpu_dict = dict()
    # for idx, geom in f1car.geometry.items():
    #     f1car_vertex_list_dict[idx] = tm.rendering.mesh_to_vertexlist(geom)
    #     f1car_gpu_dict[idx] = flipper_pipeline.vertex_list_indexed(
    #         len(f1car_vertex_list_dict[idx][4][1]) // 3, GL.GL_TRIANGLES, f1car_vertex_list_dict[idx][3]
    #     )
    #     f1car_gpu_dict[idx].position[:] = f1car_vertex_list_dict[idx][4][1]
        
    #     f1car_gpu_dict[idx].normal[:] = f1car_vertex_list_dict[idx][5][1]
    #     print("expected UV size:", len(f1car_gpu_dict[idx].uv))
    #     print("actual   UV size:", len(f1car_vertex_list_dict[idx][6][1]))
    #     f1car_gpu_dict[idx].uv[:] = f1car_vertex_list_dict[idx][6][1][:4384] # TODO Eliminate halving
    #     f1car_gpu_dict[idx].texture = texture_2D_setup(geom.visual.material.image)
        

    # TODO
    # f1car_vertex_list = tm.rendering.mesh_to_vertexlist(f1car)
    # f1car_gpu = flipper_pipeline.vertex_list_indexed(
    #     len(f1car_vertex_list[4][1]) // 3, GL.GL_TRIANGLES, f1car_vertex_list[3]
    # )
    # f1car_gpu.position[:] = f1car_vertex_list[4][1]
    
    # f1car_gpu.normal[:] = f1car_vertex_list[5][1]
    # print("expected UV size:", len(f1car_gpu.uv))
    # print("actual   UV size:", len(f1car_vertex_list[6][1]))
    # f1car_gpu.uv[:] = f1car_vertex_list[6][1][:4384] # TODO Eliminate halving
    # f1car_gpu.texture = texture_2D_setup(f1car.visual.material.image)
    
    # ===============================================================================
    # ===============================================================================
        
    
    # tendremos dos transformaciones distintas, una por conejo
    # flipper/izq|der son una matriz para conseguir la rotacion por multiplicacion @
    window.program_state = {
        "world": {
            #"space": pymunk.Space(),
            "gravity": (0, .2),
            "ground": {
                #"segment": None,
                "friction": .1,
            },
            "static_borders": []
        },
        "space": {
            "root": tr.identity(),
        },
        "flipper_floor": {
            "pmk": {
                "body": None,
                "shape": None
            },
            "root": (
                tr.identity()
            ),
            "draw": (
                tr.identity()
            ),
        },
        "flipper_ball": {
            "pmk": {
                "body": None,
                "shape": None
            },
            "root": (
                tr.identity()
            ),
            "draw": (
                tr.identity()
            ),
        },
        "flipper_walls_tall": {
            "pmk": {
                "body": None,
                "shape": None
            },
            "root": (
                tr.identity()
            ),
            "draw": (
                tr.identity()
            ),
        },
        "flipper_mushroom": {
            "pmk": {
                "body": {
                    0: None,
                    1: None,
                    2: None,
                },
                "shape": {
                    0: None,
                    1: None,
                    2: None,
                },
            },
            "root": {
                0: tr.identity(),
                1: tr.identity(),
                2: tr.identity(),
            },
            "draw": {
                0: tr.identity(),
                1: tr.identity(),
                2: tr.identity(),
            },
        },
        "flipper": {
            "pmk": {
                "body": {
                    "izq": None,
                    "der": None,
                    "up": None,
                },
                "shape": {
                    "izq": None,
                    "der": None,
                    "up": None,
                },
            },
            "root": {
                "izq": tr.identity(),
                "der": tr.identity(),
                "up": tr.identity(),
            },
            "draw": {
                "izq": tr.identity(),
                "der": tr.identity(),
                "up": tr.identity(),
            },
            "state": { # TODO: Adjust system
                "izq": {
                    "rotate": 0, # -1: down, 0: still, -1: up, 
                    "ang_off": 9,
                    "pressed": False, # Obsolete
                    "ang": 0, # Obsolete
                    "range": np.pi*3/8,
                },
                "der": {
                    "rotate": 0, # -1: down, 0: still, -1: up, 
                    "ang_off": 9,
                    "pressed": False,
                    "ang": 0, # % de giro actual
                    "range": np.pi*3/8,
                },
                "up": {
                    "rotate": 0, # -1: down, 0: still, -1: up, 
                    "ang_off": 11,
                    "pressed": False,
                    "ang": 0, # % de giro actual
                    "range": np.pi/2,
                },
            }
        },
        "total_time": 0.0,
        "transform": tr.uniformScale(2),
        "view": {
            "vec3": np.array([0.0, -0.8, -0.5]),
            "vector": tr.lookAt(
                        np.array([0.0, -0.8, -0.5]),  # posición de la cámara
                        np.array([0.0, 0.0, 0.0]),  # hacia dónde apunta
                        np.array([0.0, 1.0, 0.0]),  # vector para orientarla (arriba)
                    ),
            "state": {
                "off": 1,
                "delta": 0,
                "demo": False,
                "delta_demo": .5,
            },
        },
        "f1car": {
            "root": {
                0: tr.identity(),
                1: tr.identity(),
            },
            "draw": {
                0: tr.identity(),
                1: tr.identity(),
            },
        },
        
        "projection_type": 0, # 0: Perspectiva, 1: Orthographic
        "projection_per": tr.perspective(60, float(window_width)/float(window_height), .1, 10),
        "projection_ort": tr.ortho(-1, 1, -1, 1, .1, 200),
    }
    
    # Defining the world
    def set_world():
        global world
        
        world = pymunk.Space() # window.program_state["world"]["space"]
        world.gravity = window.program_state["world"]["gravity"]
        # Ground in this situation would be the bottom of the flipper
        # TODO: Adjust values
        groundBd = pymunk.Segment(
            world.static_body,
            (-50, -2.5),
            (50, -2.5),
            1.0
        )
        groundBd.friction = window.program_state["world"]["ground"]["friction"]
        
        world.add(groundBd)
        
        # Adding all borders
        def add_static_to_wrld(a,b, elasticity = .5, group = 1):
            line = pymunk.Segment(world.static_body, a, b, .1)
            line.elasticity = 0.7
            line.group = 1
            world.add(line)
            return line
        
        # TODO: Add borders
        add_static_to_wrld((-1,1),(1,1))
        add_static_to_wrld((1,1),(1,-1))
        add_static_to_wrld((1,-1),(-1,-1))
        add_static_to_wrld((-1,-1),(-1,1))
        
        # TODO: Ball
        # Adding ball
        ball_mass = 0.1
        ball_rad = 0.02
        ball_pmk_bd = pymunk.Body(
            ball_mass,
            pymunk.moment_for_circle(
                ball_mass,
                0,
                ball_rad
            )
        )
        ball_pmk_bd.position = (0,-0.7)
        ball_pmk_shape = pymunk.Circle(
            ball_pmk_bd,
            ball_rad
        )
        ball_pmk_shape.collision_type = 0
        world.add(ball_pmk_bd, ball_pmk_shape)
        
        window.program_state["flipper_ball"]["pmk"]["body"] = ball_pmk_bd
        window.program_state["flipper_ball"]["pmk"]["shape"] = ball_pmk_shape
        
        # TODO: Create body & shape for all
        # 0
        # window.program_state["space"]["pmk"] = (
        #     tr.identity()
        # )
        # 1
        # window.program_state["flipper_floor"]["root"] = (
        #     window.program_state["space"]["root"]
        #     @ tr.rotationX(np.pi/8)
        #     @ tr.identity()
        # )
        # TODO: Add walls as static segments above
        # 2
        # window.program_state["flipper_walls_tall"]["pmk"]["body"] = pymunk.Body(body_type=pymunk.Body.STATIC)
        # window.program_state["flipper_walls_tall"]["pmk"]["body"].angle    = 
        # window.program_state["flipper_walls_tall"]["pmk"]["body"].position = 
        # window.program_state["flipper_walls_tall"]["pmk"]["shape"] = pymunk.Poly(
        #     window.program_state["flipper_walls_tall"]["pmk"]["body"],
        #     vertices=[
        #         (vert[0], vert[1])
        #         for vert in 
        #     ]
        # )
        # TODO: aux_func to increase readability
        # TODO: Adjust
        # 2
        aux_mesh = tm.load("my_assets/tri_orangeFlipper.obj")
        aux_mesh.apply_scale(scale)
        #aux_mesh.apply_scale((,,))
        window.program_state["flipper"]["pmk"]["body"]["izq"] = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        window.program_state["flipper"]["pmk"]["body"]["izq"].angle    = -np.pi*3/16
        window.program_state["flipper"]["pmk"]["body"]["izq"].position = (-.15,.6)
        window.program_state["flipper"]["pmk"]["shape"]["izq"] = pymunk.Poly(
            window.program_state["flipper"]["pmk"]["body"]["izq"],
            vertices = [
                (vert[0], vert[1])
                for vert in [(0,0),(0,1),(1,1),(1,0)] # aux_mesh # [(0,0),(0,1),(1,1),(1,0)]
            ]
        )
        world.add(
            window.program_state["flipper"]["pmk"]["body"]["izq"],
            window.program_state["flipper"]["pmk"]["shape"]["izq"]
        )
        # 2
        # aux_mesh = tm.load("my_assets/tri_orangeFlipper.obj")
        # aux_mesh.apply_scale(scale)
        #aux_mesh.apply_scale((,,))
        # window.program_state["flipper_ball"]["pmk"]["body"] = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        # window.program_state["flipper_ball"]["pmk"]["body"].angle    = -np.pi*3/16
        # window.program_state["flipper_ball"]["pmk"]["body"].position = (-.15,.6)
        # window.program_state["flipper_ball"]["pmk"]["shape"] = pymunk.Poly(
        #     window.program_state["flipper_ball"]["pmk"]["body"],
        #     vertices = [
        #         (vert[0], vert[1])
        #         for vert in [(0,0),(0,1),(1,1),(1,0)] # aux_mesh # [(0,0),(0,1),(1,1),(1,0)]
        #     ]
        # )
        # world.add(
        #     window.program_state["flipper"]["pmk"]["body"]["izq"],
        #     window.program_state["flipper"]["pmk"]["shape"]["izq"]
        # )
        # (
        #     window.program_state["flipper_floor"]["root"]
        #     @ tr.translate(0,0,0.6)
        #     @ tr.translate(-.15,0,0)
        #     @ tr.rotationY(-np.pi*3/16)
        #     @ tr.scale(.8,.8,.8)
        # )
        # 2
        aux_mesh = tm.load("my_assets/tri_orangeFlipper.obj")
        aux_mesh.apply_scale(scale)
        aux_mesh.apply_scale((-1,1,1))
        window.program_state["flipper"]["pmk"]["body"]["der"] = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        window.program_state["flipper"]["pmk"]["body"]["der"].angle    = np.pi*3/16
        window.program_state["flipper"]["pmk"]["body"]["der"].position = (.15,.6)
        window.program_state["flipper"]["pmk"]["shape"]["der"] = pymunk.Poly(
            window.program_state["flipper"]["pmk"]["body"]["der"],
            vertices = [
                (vert[0], vert[1])
                for vert in [(0,0),(0,1),(1,1),(1,0)] # aux_mesh
            ]
        )
        world.add(
            window.program_state["flipper"]["pmk"]["body"]["der"],
            window.program_state["flipper"]["pmk"]["shape"]["der"]
        )
        # 2
        aux_mesh = tm.load("my_assets/tri_orangeFlipper.obj")
        aux_mesh.apply_scale(scale)
        aux_mesh.apply_scale(.5)
        window.program_state["flipper"]["pmk"]["body"]["up"] = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        window.program_state["flipper"]["pmk"]["body"]["up"].angle    = -np.pi/2
        window.program_state["flipper"]["pmk"]["body"]["up"].position = (-.3,-0.5)
        window.program_state["flipper"]["pmk"]["shape"]["up"] = pymunk.Poly(
            window.program_state["flipper"]["pmk"]["body"]["up"],
            vertices = [
                (vert[0], vert[1])
                for vert in [(0,0),(0,1),(1,1),(1,0)] # aux_mesh
            ]
        )
        world.add(
            window.program_state["flipper"]["pmk"]["body"]["up"],
            window.program_state["flipper"]["pmk"]["shape"]["up"]
        )
        # 2
        aux_mesh = tm.load("my_assets/tri_jpMushroom.obj")
        aux_mesh.apply_scale(scale)
        #aux_mesh.apply_scale(|) # TODO
        window.program_state["flipper_mushroom"]["pmk"]["body"][0] = pymunk.Body(body_type=pymunk.Body.STATIC)
        #window.program_state["flipper_mushroom"]["pmk"]["body"][0].angle    = 
        window.program_state["flipper_mushroom"]["pmk"]["body"][0].position = (0,0)
        window.program_state["flipper_mushroom"]["pmk"]["shape"][0] = pymunk.Poly(
            window.program_state["flipper_mushroom"]["pmk"]["body"][0],
            vertices = [
                (vert[0], vert[1])
                for vert in [(0,0),(0,1),(1,1),(1,0)] # aux_mesh
            ]
        )
        world.add(
            window.program_state["flipper_mushroom"]["pmk"]["body"][0],
            window.program_state["flipper_mushroom"]["pmk"]["shape"][0]
        )
        # 2
        aux_mesh = tm.load("my_assets/tri_jpMushroom.obj")
        aux_mesh.apply_scale(scale)
        #aux_mesh.apply_scale(|) # TODO
        window.program_state["flipper_mushroom"]["pmk"]["body"][1] = pymunk.Body(body_type=pymunk.Body.STATIC)
        #window.program_state["flipper_mushroom"]["pmk"]["body"][1].angle    = 
        window.program_state["flipper_mushroom"]["pmk"]["body"][1].position = (-.2,-.2)
        window.program_state["flipper_mushroom"]["pmk"]["shape"][1] = pymunk.Poly(
            window.program_state["flipper_mushroom"]["pmk"]["body"][1],
            vertices = [
                (vert[0], vert[1])
                for vert in [(0,0),(0,1),(1,1),(1,0)] # aux_mesh
            ]
        )
        world.add(
            window.program_state["flipper_mushroom"]["pmk"]["body"][1],
            window.program_state["flipper_mushroom"]["pmk"]["shape"][1]
        )
        # 2
        aux_mesh = tm.load("my_assets/tri_jpMushroom.obj")
        aux_mesh.apply_scale(scale)
        #aux_mesh.apply_scale(|) # TODO
        window.program_state["flipper_mushroom"]["pmk"]["body"][2] = pymunk.Body(body_type=pymunk.Body.STATIC)
        #window.program_state["flipper_mushroom"]["pmk"]["body"][2].angle    = 
        window.program_state["flipper_mushroom"]["pmk"]["body"][2].position = (.2,-.2)
        window.program_state["flipper_mushroom"]["pmk"]["shape"][2] = pymunk.Poly(
            window.program_state["flipper_mushroom"]["pmk"]["body"][2],
            vertices = [
                (vert[0], vert[1])
                for vert in [(0,0),(0,1),(1,1),(1,0)] # aux_mesh
            ]
        )
        world.add(
            window.program_state["flipper_mushroom"]["pmk"]["body"][2],
            window.program_state["flipper_mushroom"]["pmk"]["shape"][2]
        )
        
        # ===============================================================
        window.program_state["world"]["space"] = world
        window.program_state["world"]["ground"]["segment"] = groundBd
        # ===============================================================
        
        
    # Locating all objects
    def set_roots():
        # 0
        window.program_state["space"]["root"] = (
            tr.identity()
            #@ tr.rotationY(np.pi)
        )
        # 1
        window.program_state["flipper_floor"]["root"] = (
            window.program_state["space"]["root"]
            #@ tr.rotationX(np.pi/8)
            @ tr.identity()
        )
        # 2 #TODO
        window.program_state["flipper_ball"]["root"] = (
            window.program_state["flipper_floor"]["root"]
            @ tr.identity()
            @ tr.translate(0,0.01,0.1)
        )
        # 2
        window.program_state["flipper_walls_tall"]["root"] = (
            window.program_state["flipper_floor"]["root"]
            @ tr.identity()
            @ tr.translate(0,-0.01,0)
        )
        # 2
        window.program_state["flipper"]["root"]["izq"] = (
            window.program_state["flipper_floor"]["root"]
            @ tr.translate(0,0,0.6*scale_mult)
            @ tr.translate(-.15*scale_mult,0,0)
            @ tr.rotationY(-np.pi*3/16)
            @ tr.scale(.8,.8,.8)
        )
        # 2
        window.program_state["flipper"]["root"]["der"] = (
            window.program_state["flipper_floor"]["root"]
            @ tr.translate(0,0,0.6*scale_mult)
            @ tr.translate(.15*scale_mult,0,0)
            @ tr.rotationY(np.pi*3/16)
            @ tr.scale(-.8,.8,.8)
        )
        # 2
        window.program_state["flipper"]["root"]["up"] = (
            window.program_state["flipper_floor"]["root"]
            #@ tr.translate(0,0,0)
            @ tr.translate(-.3*scale_mult,0,-0.5*scale_mult)
            @ tr.rotationY(-np.pi/2)
            @ tr.uniformScale(.5)
        )
        # 2
        window.program_state["flipper_mushroom"]["root"][0] = (
            window.program_state["flipper_floor"]["root"]
            @ tr.translate(0,0,0)
        )
        # 2
        window.program_state["flipper_mushroom"]["root"][1] = (
            window.program_state["flipper_floor"]["root"]
            @ tr.translate(-.2*scale_mult,0,-.2*scale_mult)
        )
        # 2
        window.program_state["flipper_mushroom"]["root"][2] = (
            window.program_state["flipper_floor"]["root"]
            @ tr.translate(0.2*scale_mult,0,-.2*scale_mult)
        )
        # 1
        window.program_state["f1car"]["root"][0] = (
            window.program_state["space"]["root"]
            @ tr.translate(0,-1*scale_mult,0)
        )
        # 1
        window.program_state["f1car"]["root"][1] = (
            window.program_state["space"]["root"]
            @ tr.translate(0,-1*scale_mult,0)
        )
    
    
    # =======================================================
    # Aux Funcs =============================================
    
    # Modify the eye position to rotate
    # Every time the perspective changes, the eye position resets
    def update_eye_position(view, dt, tp):
        state = view["state"]
        if tp==0:
            if state["demo"]: view["vector"] @= tr.rotationY(dt * state["delta_demo"])
            else: view["vector"] @= tr.rotationY(dt * state["delta"])
    
    # Modify angle attribute
    def rotate_flipper(state, body, dt):
        # TODO: Maybe add angle limiters once the thing is working
        # TODO: Prolly won't work
        # mult =  state["rotate"]
        mult = 1 if state["pressed"] else -1
        new_ang = state["ang"] + mult * state["ang_off"] * dt
        
        state["ang"] = min(1,max(0,new_ang))
        
        if state["ang"] == 0 or state["ang"] == 1:
            body.angular_velocity = 0
        
    
    # return the matrix corresponding to the final angle
    # TODO Check
    def flipper_rotation_matrix(state:dict):
        # s = state["ang"]
        # angle = s*state["range"]
        angle = window.program_state["flipper"]["pmk"]["body"]["up"].angle
        return tr.rotationY(angle)
    
    def get_projection():
        kw = "projection_per" if window.program_state["projection_type"] == 0 \
            else "projection_ort"
        return window.program_state[kw].reshape(
            16, 1, order="F"
        )
    
    def get_car_pos(idx,tot_num,time):
        angle = time * 2.0
        small_rad = .22
        big_rad = 1.7
        small_ang_0 = (idx)/tot_num*2*np.pi - 0.7 * angle - 1.4 * np.sin(time*1.3)
        
        return (
            tr.rotationY(angle)
            @ tr.translate(0,0,big_rad)
            @ tr.rotationY(small_ang_0)
            @ tr.translate(0,0,small_rad)
            @ tr.rotationY(-small_ang_0)
            @ tr.rotationY(np.pi+.1)
        )
    
    def get_ball_pos(ball_body): 
        x,z = ball_body.position
        return tr.translate(x,0,z)
        
        
    # =======================================================
    # Events ================================================
    
    key = pyglet.window.key
    
    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.A:
            window.program_state["flipper"]["state"]["izq"]["pressed"] = True
            window.program_state["flipper"]["state"]["up"]["pressed"] = True
            window.program_state["flipper"]["pmk"]["body"]["izq"].angular_velocity = (
                window.program_state["flipper"]["state"]["izq"]["ang_off"] * 1
            )
            window.program_state["flipper"]["pmk"]["body"]["up"].angular_velocity = (
                window.program_state["flipper"]["state"]["up"]["ang_off"] * 1
            )
        if symbol == key.D:
            window.program_state["flipper"]["state"]["der"]["pressed"] = True
            window.program_state["flipper"]["pmk"]["body"]["der"].angular_velocity = (
                window.program_state["flipper"]["state"]["der"]["ang_off"] * 1 * -1
            )
        if symbol == key.C:
            print(window.program_state["projection_type"])
            if not window.program_state["view"]["state"]["demo"]:
                if window.program_state["projection_type"]:
                    window.program_state["projection_type"] = 0
                    window.program_state["view"]["vec3"] = np.array([0.0, -0.8, -0.5])
                    window.program_state["view"]["vector"] = tr.lookAt(
                        np.array([0.0, 1.6, 1.6]),  # posición de la cámara
                        np.array([0.0, 0.0, 0.0]),  # hacia dónde apunta
                        np.array([0.0, 1.0, 0.0]),  # vector para orientarla (arriba)
                    )
                else:
                    window.program_state["projection_type"] = 1
                    window.program_state["view"]["vec3"] = np.array([0.0, 2.3, 1.0])  # posición de la cámara
                    # window.program_state["view"]["vector"] = tr.lookAt(
                    #     np.array([0.0, 2.3, 1.0]),  # posición de la cámara
                    #     np.array([0.0, 0.0, 0.0]),  # hacia dónde apunta
                    #     np.array([0.0, 1.0, 0.0]),  # vector para orientarla (arriba)
                    # )
                    window.program_state["view"]["vector"] = tr.lookAt(
                        np.array([0.0, 2.3, 1.0]),  # posición de la cámara
                        np.array([0.0, 0.0, 0.0]),  # hacia dónde apunta
                        np.array([0.0, 1.0, 0.0]),  # vector para orientarla (arriba)
                    )
        # if symbol == key.Q:
        #     window.program_state["view"]["state"]["delta"] += window.program_state["view"]["state"]["off"]
        # if symbol == key.E:
        #     window.program_state["view"]["state"]["delta"] -= window.program_state["view"]["state"]["off"]
        if symbol == key.SPACE:
            window.program_state["flipper_ball"]["pmk"]["body"].position = (0,0)
        if symbol == key.P:
            pprint.pprint(window.program_state["flipper_floor"]["root"])
            
            # if not window.program_state["projection_type"]:
            #     window.program_state["view"]["state"]["demo"] = not window.program_state["view"]["state"]["demo"]
            
    
    @window.event
    def on_key_release(symbol, modifiers):
        if symbol == key.A:
            window.program_state["flipper"]["state"]["izq"]["pressed"] = False
            window.program_state["flipper"]["state"]["up"]["pressed"] = False
            window.program_state["flipper"]["pmk"]["body"]["izq"].angular_velocity = (
                window.program_state["flipper"]["state"]["izq"]["ang_off"] * -1
            )
            window.program_state["flipper"]["pmk"]["body"]["up"].angular_velocity = (
                window.program_state["flipper"]["state"]["up"]["ang_off"] * -1
            )
        if symbol == key.D:
            window.program_state["flipper"]["state"]["der"]["pressed"] = False
            window.program_state["flipper"]["pmk"]["body"]["der"].angular_velocity = (
                window.program_state["flipper"]["state"]["der"]["ang_off"] * -1 * -1
            )
        # if symbol == key.Q:
        #     window.program_state["view"]["state"]["delta"] -= window.program_state["view"]["state"]["off"]
        # if symbol == key.E:
        #     window.program_state["view"]["state"]["delta"] += window.program_state["view"]["state"]["off"]
    
    
    @window.event
    def on_draw():
        
        GL.glClearColor(0.0, 0.6, 0.5, 1.0)
        GL.glLineWidth(1.0)
        #GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_DEPTH_TEST)
        # GL.glViewport(0, 0, *window.size)
        # GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        window.clear()
        
        # TODO fix draw mode
        # dibujamos nuestro segundo objeto. usamos el wireframe (GL_LINE)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        flipper_pipeline.use()
        
        # TODO add all required variables
        flipper_pipeline["lightPosition1"] = np.array([ 3, 3, -7])
        flipper_pipeline["lightPosition2"] = np.array([-3, 3, -7])
        
        flipper_pipeline["projection"] = get_projection()
        # print(flipper_pipeline["projection"])
        flipper_pipeline["viewPosition"] = window.program_state["view"]["vec3"]
        flipper_pipeline["view"] = window.program_state["view"]["vector"].reshape(
            16, 1, order="F"
        )
        
        
        # The floor of the flipper
        transform = window.program_state["flipper_floor"]["draw"]
        flipper_pipeline["transform"] = transform.reshape(16, 1, order="F")
        
        # ===============================================================
        GL.glBindTexture(GL.GL_TEXTURE_2D, flipper_floor_gpu.texture)
        for attr in ["Kd", "Ks", "Ka"]:
            flipper_pipeline[attr] = flipper_floor_gpu.__dict__[attr][:3] / 255
        flipper_pipeline["shininess"] = flipper_floor_gpu.ns / 255
        # ===============================================================
        flipper_floor_gpu.draw(pyglet.gl.GL_TRIANGLES)
        
        
        # 
        transform = window.program_state["flipper_walls_tall"]["draw"]
        flipper_pipeline["transform"] = transform.reshape(16, 1, order="F")
        GL.glBindTexture(GL.GL_TEXTURE_2D, flipper_walls_tall_gpu.texture)
        for attr in ["Kd", "Ks", "Ka"]:
            flipper_pipeline[attr] = flipper_walls_tall_gpu.__dict__[attr][:3] / 255
        flipper_pipeline["shininess"] = flipper_walls_tall_gpu.ns / 255
        flipper_walls_tall_gpu.draw(pyglet.gl.GL_TRIANGLES)
        
        
        # BALL
        transform = window.program_state["flipper_ball"]["draw"]
        flipper_pipeline["transform"] = transform.reshape(16, 1, order="F")
        GL.glBindTexture(GL.GL_TEXTURE_2D, flipper_ball_gpu.texture)
        for attr in ["Kd", "Ks", "Ka"]:
            flipper_pipeline[attr] = flipper_ball_gpu.__dict__[attr][:3] / 255
        flipper_pipeline["shininess"] = flipper_ball_gpu.ns / 255
        flipper_ball_gpu.draw(pyglet.gl.GL_TRIANGLES)
        
        
        
        for transform in window.program_state["flipper_mushroom"]["draw"].values():
            flipper_pipeline["transform"] = transform.reshape(16, 1, order="F")
            GL.glBindTexture(GL.GL_TEXTURE_2D, flipper_mushroom_gpu.texture)
            for attr in ["Kd", "Ks", "Ka"]:
                flipper_pipeline[attr] = flipper_mushroom_gpu.__dict__[attr][:3] / 255
            flipper_pipeline["shininess"] = flipper_mushroom_gpu.ns / 255
            flipper_mushroom_gpu.draw(pyglet.gl.GL_TRIANGLES)
        
        # TODO
        # for transform in window.program_state["f1car"]["draw"].values():
        #     flipper_pipeline["transform"] = transform.reshape(16, 1, order="F")
        #     for attr in ["Kd", "Ks", "Ka"]:
        #         flipper_pipeline[attr] = f1car_gpu.__dict__[attr][:3] / 255
        #     flipper_pipeline["shininess"] = f1car_gpu.ns / 255
        #     f1car_gpu.draw(pyglet.gl.GL_TRIANGLES)
            # for idx in f1car_gpu_dict.keys():
            #     for attr in ["Kd", "Ks", "Ka"]:
            #         flipper_pipeline[attr] = f1car_gpu_dict[idx].__dict__[attr][:3] / 255
            #     flipper_pipeline["shininess"] = f1car_gpu_dict[idx].ns / 255
            #     f1car_gpu_dict[idx].draw(pyglet.gl.GL_TRIANGLES)
            
        
        for transform in window.program_state["flipper"]["draw"].values():
            # flipper_pipeline["view_transform"] = transform.reshape(16, 1, order="F")
            flipper_pipeline["transform"] = transform.reshape(16, 1, order="F")
            GL.glBindTexture(GL.GL_TEXTURE_2D, flipper_gpu.texture)
            for attr in ["Kd", "Ks", "Ka"]:
                flipper_pipeline[attr] = flipper_gpu.__dict__[attr][:3] / 255
            flipper_pipeline["shininess"] = flipper_gpu.ns / 255
            flipper_gpu.draw(pyglet.gl.GL_TRIANGLES)
        
    
    def update_world(dt, window):
        window.program_state["total_time"] += dt
        total_time = window.program_state["total_time"]
        
        update_eye_position(window.program_state["view"], dt, window.program_state["projection_type"])
        
        # Here to transform
        window.program_state["flipper_floor"]["draw"] = window.program_state["flipper_floor"]["root"]
        window.program_state["flipper_walls_tall"]["draw"] = window.program_state["flipper_walls_tall"]["root"]
        
        # TODO update ball pos
        # print(window.program_state["flipper_ball"]["pmk"]["body"].position)
        window.program_state["flipper_ball"]["draw"] = (
            window.program_state["flipper_ball"]["root"]
            @ get_ball_pos(window.program_state["flipper_ball"]["pmk"]["body"])
        )
        
        for key in window.program_state["flipper_mushroom"]["draw"].keys():
            window.program_state["flipper_mushroom"]["draw"][key] = window.program_state["flipper_mushroom"]["root"][key]
        
        # TODO
        # for key in window.program_state["f1car"]["draw"].keys():
        #     window.program_state["f1car"]["draw"][key] = (
        #         window.program_state["f1car"]["root"][key]
        #         @ get_car_pos(
        #             key,
        #             max(*window.program_state["f1car"]["draw"].keys()) + 1,
        #             total_time)
        #         )
        
        for side in window.program_state["flipper"]["state"].keys():
            rotate_flipper(
                window.program_state["flipper"]["state"][side],
                window.program_state["flipper"]["pmk"]["body"][side],
                dt)
            
            window.program_state["flipper"]["draw"][side] = (
                window.program_state["flipper"]["root"][side]
                @ flipper_rotation_matrix(
                        window.program_state["flipper"]["state"][side]
                    )
            )
            
        world.step(dt)
        
        
    # Set all
    set_world()
    set_roots()
    
    pyglet.clock.schedule_interval(update_world, 1 / 60.0, window)
    pyglet.app.run(1 / 60.0)
    