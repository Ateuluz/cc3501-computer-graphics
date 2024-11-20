import os.path
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pyglet
import pyglet.gl as GL
import trimesh as tm
from pyglet.window import key
import pymunk

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
)

if sys.path[0] != '':
    sys.path.insert(0, '')

import grafica.transformations as tr
from grafica.textures import texture_2D_setup

# Se crea el grafo de escena

def create_pinball(meshBall, meshPinball, meshFlipper, meshObstaculo1, meshObstaculo2, meshDestroyer, meshXwing):
    graph = nx.DiGraph(root = "pinball")

    

    graph.add_node(
        "pinball",
        name = "pinball",
        transform = tr.rotationZ(-np.pi/2)
        )
    graph.add_node(
        "pinball_geometry",
        name = "pinball_geometry", 
        mesh = meshPinball,
        transform = tr.uniformScale(2.1))
    
    graph.add_node(
        "ball",
        name = "ball",
        transform = tr.translate(0, 0, .16)
        )
    
    graph.add_node(
        "ball_geometry",
        name = "ball_geometry",
        mesh = meshBall,
        transform = tr.uniformScale(0.2)
        )
    graph.add_edge("pinball", "ball")
    graph.add_edge("ball", "ball_geometry")
    graph.add_edge("pinball", "pinball_geometry")

    graph.add_node(
        "obstaculo1",
        name = "obstaculo1",
        transform = tr.translate(0, -.4, .11) @ tr.rotationX(-np.pi/2))
    graph.add_node(
        "obstaculo1_geometry",
        name = "obstaculo1_geometry",
        mesh = meshObstaculo1,
        transform = tr.uniformScale(.25))

    graph.add_edge("pinball", "obstaculo1")
    graph.add_edge("obstaculo1", "obstaculo1_geometry")

    graph.add_node(
        "obstaculo1.2",
        name = "obstaculo1.2",
        transform = tr.translate(0, .4, .11) @ tr.rotationX(-np.pi/2))
    graph.add_node(
        "obstaculo1.2_geometry",
        name = "obstaculo1.2_geometry",
        mesh = meshObstaculo1,
        transform = tr.uniformScale(.25))

    graph.add_edge("pinball", "obstaculo1.2")
    graph.add_edge("obstaculo1.2", "obstaculo1.2_geometry")


    graph.add_node(
        "obstaculo2",
        name = "obstaculo2",
        transform = tr.translate(-0.75, 0, .175) @ tr.rotationZ(3*np.pi/4) @ tr.rotationX(np.pi/2))
    graph.add_node(
        "obstaculo2_geometry",
        name = "obstaculo2_geometry",
        mesh = meshObstaculo2,
        transform = tr.uniformScale(.6))

    graph.add_edge("pinball", "obstaculo2")
    graph.add_edge("obstaculo2", "obstaculo2_geometry")

    graph.add_node(
        "Destroyer",
        name = "Destroyer",
        transform = tr.translate(2.5,0,0))
    graph.add_node(
        "Destroyer_geometry",
        name = "Destroyer_geometry",
        mesh = meshDestroyer,
        transform = tr.uniformScale(0.3))
    
    graph.add_edge("pinball", "Destroyer")
    graph.add_edge("Destroyer", "Destroyer_geometry")

    graph.add_node(
        "X-Wing",
        name = "X-Wing",
        transform = tr.translate(-2.5, 0, 0) @ tr.rotationZ(np.pi/2))
    graph.add_node(
        "x-wing_geometry",
        name = "x-wing_geometry",
        mesh = meshXwing,
        transform = tr.uniformScale(0.3))
    
    graph.add_edge("pinball", "X-Wing")
    graph.add_edge("X-Wing", "x-wing_geometry")

    graph.add_node(
        "flipper_der",
        name = "flipper_der",
        transform = tr.translate(1.07, .55, .125) @ tr.scale(1.0, -1.0, 1.0))
    graph.add_node(
        "flipper_der_geometry",
        name = "flipper_der_geometry",
        mesh = meshFlipper,
        transform = tr.uniformScale(0.3))
    
    graph.add_edge("pinball", "flipper_der")
    graph.add_edge("flipper_der", "flipper_der_geometry")

    graph.add_node(
        "flipper_izq",
        name = "flipper_izq",
        transform = tr.translate(1.07, -.55, 0.125))
    graph.add_node(
        "flipper_izq_geometry",
        name = "flipper_izq_geometry",
        mesh = meshFlipper,
        transform = tr.uniformScale(0.3))
    
    graph.add_edge("pinball", "flipper_izq")
    graph.add_edge("flipper_izq", "flipper_izq_geometry")

    return graph

def update_pinball(dt, window):
    window.program_state["total_time"] += dt
    total_time = window.program_state["total_time"]

    for a in range(10):
        world.step(dt/10)

    graph = window.program_state["scene_graph"]

    graph.nodes["Destroyer"]["transform"] = tr.rotationZ(-total_time*2.5) @ tr.translate(2.5, 0, 0)
    graph.nodes["X-Wing"]["transform"] = tr.rotationZ(total_time*3.5) @ tr.translate(-2.5, 0, 1) @ tr.rotationZ(-np.pi/2)

if __name__ == "__main__":

    width = 960
    height = 960

    window = pyglet.window.Window(width, height)

    meshBall = tm.load("Tarea2/assets/BallModel/Ball.obj")
    model_scale_ball = tr.uniformScale(2.0 / meshBall.scale)
    model_translate_ball = tr.translate(*-meshBall.centroid)
    meshBall.apply_transform(model_scale_ball @ model_translate_ball)

    meshPinball = tm.load("Tarea2/assets/pinballModels/Pinball.obj", force="mesh")
    model_scale_pinball = tr.uniformScale(2.0 / meshPinball.scale)
    model_translate_pinball = tr.translate(*-meshPinball.centroid)
    model_rotate_pinball = tr.rotationX(np.pi/2)
    meshPinball.apply_transform(model_rotate_pinball @ model_scale_pinball @ model_translate_pinball)


    meshObstaculo1 = tm.load("Tarea2/assets/pinballModels/obstaculo1.obj", force="mesh")
    model_scale_obstaculo1 = tr.uniformScale(2.0 / meshObstaculo1.scale)
    model_translate_obstaculo1 = tr.translate(*-meshObstaculo1.centroid)
    model_rotate_obstaculo1 = tr.rotationX(np.pi/2)
    meshObstaculo1.apply_transform(model_rotate_obstaculo1 @ model_scale_obstaculo1 @ model_translate_obstaculo1)
    

    meshObstaculo2 = tm.load("Tarea2/assets/pinballModels/obstaculo2.obj", force="mesh")
    model_scale_obstaculo2 = tr.uniformScale(2.0 / meshObstaculo2.scale)
    model_translate_obstaculo2 = tr.translate(*-meshObstaculo2.centroid)
    meshObstaculo2.apply_transform(model_scale_obstaculo2 @ model_translate_obstaculo2)

    meshFlipper = tm.load("Tarea2/assets/pinballModels/flipper.obj", force="mesh")
    model_scale_flipper = tr.uniformScale(2.0 / meshFlipper.scale)
    model_rotate_flipper = tr.rotationX(np.pi/2) @ tr.rotationY(-np.pi/4)
    meshFlipper.apply_transform(model_scale_flipper @ model_rotate_flipper)
    
    meshXwing = tm.load("Tarea2/assets/ShipModels/xWing.obj", force="mesh")
    model_scale_xwing = tr.uniformScale(2.0 / meshXwing.scale)
    model_translate_xwing = tr.translate(*-meshXwing.centroid)
    model_rotate_xwing = tr.rotationX(np.pi/2)
    meshXwing.apply_transform(model_scale_xwing @ model_translate_xwing @ model_rotate_xwing)

    meshTieFighter = tm.load("Tarea2/assets/ShipModels/tieFighter.obj", force="mesh")
    model_scale_tiefighter = tr.uniformScale(2.0 / meshTieFighter.scale)
    meshTieFighter.apply_transform(model_scale_tiefighter)


    with open(Path(os.path.dirname(__file__)) / "Tarea2/tex_vertex_program.glsl") as f:
        vertex_source_code = f.read()
    
    with open(Path(os.path.dirname(__file__)) / "Tarea2/tex_fragment_program.glsl") as f:
        fragment_source_code = f.read()
    
    vert_shader = pyglet.graphics.shader.Shader(vertex_source_code, "vertex")
    frag_shader = pyglet.graphics.shader.Shader(fragment_source_code, "fragment")
    tex_pipeline = pyglet.graphics.shader.ShaderProgram(vert_shader,frag_shader)
    tex_pipeline["light_position"] = np.array([0.0, -10.0, 10.0])

    ball_vertex_list = tm.rendering.mesh_to_vertexlist(meshBall)
    ball_mesh_gpu = tex_pipeline.vertex_list_indexed(
        len(ball_vertex_list[4][1]) // 3,
        GL.GL_TRIANGLES,
        ball_vertex_list[3]
    )
    ball_mesh_gpu.position[:] = ball_vertex_list[4][1]
    ball_mesh_gpu.uv[:] = ball_vertex_list[6][1]
    ball_mesh_gpu.normal[:] = ball_vertex_list[5][1]
    ball_mesh_gpu.texture = texture_2D_setup(meshBall.visual.material.image)
    for attr in meshBall.visual.material.__dict__:
        #print(f"{attr} = {meshPinball.visual.material.__dict__[attr]}")
        match attr:
            case "ambient":
                #print(f"{attr} = ambient")
                ball_mesh_gpu.Ka = meshBall.visual.material.__dict__[attr]
            case "diffuse":
                #print(f"{attr} = diffuse")
                ball_mesh_gpu.Kd = meshBall.visual.material.__dict__[attr]
            case "specular":
                #print(f"{attr} = specular")
                ball_mesh_gpu.Ks = meshBall.visual.material.__dict__[attr]
            case "kwargs":
                ball_mesh_gpu.ns = meshBall.visual.material.__dict__[attr]["ns"]
                #print(f"{meshPinball.visual.material.__dict__[attr]["ns"]} = ns")

    pinball_vertex_list = tm.rendering.mesh_to_vertexlist(meshPinball)
    pinball_mesh_gpu = tex_pipeline.vertex_list_indexed(
        len(pinball_vertex_list[4][1]) // 3,
        GL.GL_TRIANGLES,
        pinball_vertex_list[3]
    )
    pinball_mesh_gpu.position[:] = pinball_vertex_list[4][1]
    pinball_mesh_gpu.uv[:] = pinball_vertex_list[6][1]
    pinball_mesh_gpu.normal[:] = pinball_vertex_list[5][1]
    pinball_mesh_gpu.texture = texture_2D_setup(meshPinball.visual.material.image)
    for attr in meshPinball.visual.material.__dict__:
        #print(f"{attr} = {meshPinball.visual.material.__dict__[attr]}")
        match attr:
            case "ambient":
                #print(f"{attr} = ambient")
                pinball_mesh_gpu.Ka = meshPinball.visual.material.__dict__[attr]
            case "diffuse":
                #print(f"{attr} = diffuse")
                pinball_mesh_gpu.Kd = meshPinball.visual.material.__dict__[attr]
            case "specular":
                #print(f"{attr} = specular")
                pinball_mesh_gpu.Ks = meshPinball.visual.material.__dict__[attr]
            case "kwargs":
                pinball_mesh_gpu.ns = meshPinball.visual.material.__dict__[attr]["ns"]
                #print(f"{meshPinball.visual.material.__dict__[attr]["ns"]} = ns")
            
    obstaculo1_vertex_list = tm.rendering.mesh_to_vertexlist(meshObstaculo1)
    obstaculo1_mesh_gpu = tex_pipeline.vertex_list_indexed(
        len(obstaculo1_vertex_list[4][1]) // 3,
        GL.GL_TRIANGLES,
        obstaculo1_vertex_list[3]
    )
    obstaculo1_mesh_gpu.position[:] = obstaculo1_vertex_list[4][1]
    obstaculo1_mesh_gpu.uv[:] = obstaculo1_vertex_list[6][1]
    obstaculo1_mesh_gpu.normal[:] = obstaculo1_vertex_list[5][1]
    obstaculo1_mesh_gpu.texture = texture_2D_setup(meshObstaculo1.visual.material.image)
    for attr in meshObstaculo1.visual.material.__dict__:
        #print(f"{attr} = {meshObstaculo1.visual.material.__dict__[attr]}")
        match attr:
            case "ambient":
                #print(f"{attr} = ambient")
                obstaculo1_mesh_gpu.Ka = meshObstaculo1.visual.material.__dict__[attr]
            case "diffuse":
                #print(f"{attr} = diffuse")
                obstaculo1_mesh_gpu.Kd = meshObstaculo1.visual.material.__dict__[attr]
            case "specular":
                #print(f"{attr} = specular")
                obstaculo1_mesh_gpu.Ks = meshObstaculo1.visual.material.__dict__[attr]
            case "kwargs":
                obstaculo1_mesh_gpu.ns = meshObstaculo1.visual.material.__dict__[attr]["ns"]
                #print(f"{meshObstaculo1.visual.material.__dict__[attr]["ns"]} = ns")

    obstaculo2_vertex_list = tm.rendering.mesh_to_vertexlist(meshObstaculo2)
    obstaculo2_mesh_gpu = tex_pipeline.vertex_list_indexed(
        len(obstaculo2_vertex_list[4][1]) // 3,
        GL.GL_TRIANGLES,
        obstaculo2_vertex_list[3]
    )
    obstaculo2_mesh_gpu.position[:] = obstaculo2_vertex_list[4][1]
    obstaculo2_mesh_gpu.uv[:] = obstaculo2_vertex_list[6][1]
    obstaculo2_mesh_gpu.normal[:] = obstaculo2_vertex_list[5][1]
    obstaculo2_mesh_gpu.texture = texture_2D_setup(meshObstaculo2.visual.material.image)
    for attr in meshObstaculo2.visual.material.__dict__:
        #print(f"{attr} = {meshObstaculo2.visual.material.__dict__[attr]}")
        match attr:
            case "ambient":
                #print(f"{attr} = ambient")
                obstaculo2_mesh_gpu.Ka = meshObstaculo2.visual.material.__dict__[attr]
            case "diffuse":
                #print(f"{attr} = diffuse")
                obstaculo2_mesh_gpu.Kd = meshObstaculo2.visual.material.__dict__[attr]
            case "specular":
                #print(f"{attr} = specular")
                obstaculo2_mesh_gpu.Ks = meshObstaculo2.visual.material.__dict__[attr]
            case "kwargs":
                obstaculo2_mesh_gpu.ns = meshObstaculo2.visual.material.__dict__[attr]["ns"]
                #print(f"{meshObstaculo2.visual.material.__dict__[attr]["ns"]} = ns")

    flipper_vertex_list = tm.rendering.mesh_to_vertexlist(meshFlipper)
    flipper_mesh_gpu = tex_pipeline.vertex_list_indexed(
        len(flipper_vertex_list[4][1]) // 3,
        GL.GL_TRIANGLES,
        flipper_vertex_list[3]
    )
    flipper_mesh_gpu.position[:] = flipper_vertex_list[4][1]
    print(len(flipper_mesh_gpu.uv), 2*len(flipper_vertex_list[4][1])//3)
    flipper_mesh_gpu.uv[:] = flipper_vertex_list[6][1]
    flipper_mesh_gpu.normal[:] = flipper_vertex_list[5][1]
    flipper_mesh_gpu.texture = texture_2D_setup(meshFlipper.visual.material.image)
    for attr in meshFlipper.visual.material.__dict__:
        #print(f"{attr} = {meshFlipper.visual.material.__dict__[attr]}")
        match attr:
            case "ambient":
                #print(f"{attr} = ambient")
                flipper_mesh_gpu.Ka = meshFlipper.visual.material.__dict__[attr]
            case "diffuse":
                #print(f"{attr} = diffuse")
                flipper_mesh_gpu.Kd = meshFlipper.visual.material.__dict__[attr]
            case "specular":
                #print(f"{attr} = specular")
                flipper_mesh_gpu.Ks = meshFlipper.visual.material.__dict__[attr]
            case "kwargs":
                flipper_mesh_gpu.ns = meshFlipper.visual.material.__dict__[attr]["ns"]
                #print(f"{meshFlipper.visual.material.__dict__[attr]["ns"]} = ns")

    xwing_vertex_list = tm.rendering.mesh_to_vertexlist(meshXwing)
    xwing_mesh_gpu = tex_pipeline.vertex_list_indexed(
        len(xwing_vertex_list[4][1]) // 3,
        GL.GL_TRIANGLES,
        xwing_vertex_list[3]
    )
    xwing_mesh_gpu.position[:] = xwing_vertex_list[4][1]
    print(len(xwing_mesh_gpu.uv), len(xwing_vertex_list[6][1]))
    xwing_mesh_gpu.uv[:] = xwing_vertex_list[6][1][:205140]
    xwing_mesh_gpu.normal[:] = xwing_vertex_list[5][1]
    xwing_mesh_gpu.texture = texture_2D_setup(meshXwing.visual.material.image)
    for attr in meshXwing.visual.material.__dict__:
        match attr:
            case "ambient":
                xwing_mesh_gpu.Ka = meshXwing.visual.material.__dict__[attr]
            case "diffuse":
                xwing_mesh_gpu.Kd = meshXwing.visual.material.__dict__[attr]
            case "specular":
                xwing_mesh_gpu.Ks = meshXwing.visual.material.__dict__[attr]
            case "kwargs":
                xwing_mesh_gpu.ns = meshXwing.visual.material.__dict__[attr]["ns"]

    tiefighter_vertex_list = tm.rendering.mesh_to_vertexlist(meshTieFighter)
    tiefighter_mesh_gpu = tex_pipeline.vertex_list_indexed(
        len(tiefighter_vertex_list[4][1]) // 3,
        GL.GL_TRIANGLES,
        tiefighter_vertex_list[3]
    )
    tiefighter_mesh_gpu.position[:] = tiefighter_vertex_list[4][1]
    print(len(tiefighter_mesh_gpu.uv), len(tiefighter_vertex_list[6][1])/2)
    tiefighter_mesh_gpu.uv[:] = tiefighter_vertex_list[6][1][:21174]
    tiefighter_mesh_gpu.normal[:] = tiefighter_vertex_list[5][1]
    tiefighter_mesh_gpu.texture = texture_2D_setup(meshTieFighter.visual.material.image)
    for attr in meshTieFighter.visual.material.__dict__:
        match attr:
            case "ambient":
                tiefighter_mesh_gpu.Ka = meshTieFighter.visual.material.__dict__[attr]
            case "diffuse":
                tiefighter_mesh_gpu.Kd = meshTieFighter.visual.material.__dict__[attr]
            case "specular":
                tiefighter_mesh_gpu.Ks = meshTieFighter.visual.material.__dict__[attr]
            case "kwargs":
                tiefighter_mesh_gpu.ns = meshTieFighter.visual.material.__dict__[attr]["ns"]

    graph = create_pinball(ball_mesh_gpu,pinball_mesh_gpu,flipper_mesh_gpu,obstaculo1_mesh_gpu,obstaculo2_mesh_gpu,tiefighter_mesh_gpu,xwing_mesh_gpu)

    # Se crea el mundo de pymunk
    world = pymunk.Space()
    world.gravity = (0, -1)

    

    # Paredes
    static_lines = [
        # pymunk.Segment(world.static_body, (-10, 10), (10, -10), .001),
        pymunk.Segment(world.static_body, (-4.5, -10), (-4.5, 10), .1), # Pared izq
        pymunk.Segment(world.static_body, (4.5, -10), (4.5, 10), .1), # Pared Der
        pymunk.Segment(world.static_body, (-10, -8.5), (10, -8.5), .1), # Fondo abajo
        pymunk.Segment(world.static_body, (-4.5, 7.5), (4.5, 7.5), .1), # Pared arriba
        pymunk.Segment(world.static_body, (2, 7.5), (3.25, 6.5), .1), # Curva 1 derecha
        pymunk.Segment(world.static_body, (3.25, 6.5), (4.5, 3.5), .1), # Curva 2 derecha
        pymunk.Segment(world.static_body, (-2, 7.5), (-3.25, 6.5), .1), # Curva 1 izquierda
        pymunk.Segment(world.static_body, (-3.25, 6.5), (-4.5, 3.5), .1), # Curva 2 izquierda
        pymunk.Segment(world.static_body, (4.5, -4.5), (0, -9), .1), # Esquina Abajo derecha
        pymunk.Segment(world.static_body, (-4.5, -4.5), (0, -9), .1), # Esquina Abajo izquierda

        # Obstaculo 2

        pymunk.Segment(world.static_body, (-2.5, 3.8), (0, 4.2), .1), # Arriba izquierda
        pymunk.Segment(world.static_body, (0, 4.2), (2.5, 3.8), .1), # Arriba derecha
        pymunk.Segment(world.static_body, (-2.5, 3.8), (0, 3), .1), # Abajo izquierda
        pymunk.Segment(world.static_body, (2.5, 3.8), (0, 3), .1), # Abajo derecha
    ]
    for line in static_lines:
        line.elasticity = 0.7
        line.group = 1
    world.add(*static_lines)

    ball_mass = 0.142
    ball_body = pymunk.Body(ball_mass, pymunk.moment_for_circle(ball_mass, 0.35, 0.35, (0,0)))

    ball_body.position = (0, 0)

    radio = .7
    ball_shape = pymunk.Circle(ball_body, radio, (0,0))
    
    world.add(ball_body, ball_shape)

    vertices = meshFlipper.apply_scale(1.4).vertices
    vertices1 = [(v[0], v[1]) for v in vertices]
    flipper_der_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    flipper_der_body.angle = np.pi/4
    flipper_der_shape = pymunk.Poly(flipper_der_body, vertices=vertices1)
    flipper_der_body.position = (2.7500000596046448, -5.350000262260437)

    world.add(flipper_der_body, flipper_der_shape)

    vertices = meshFlipper.apply_scale(0.9).vertices
    vertices2 = [(v[0], v[1]) for v in vertices]
    flipper_izq_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    flipper_izq_body.angle = 3*np.pi/4
    flipper_izq_shape = pymunk.Poly(flipper_izq_body, vertices=vertices2)
    flipper_izq_body.position = (-2.7500000596046448, -5.350000262260437)

    world.add(flipper_izq_body, flipper_izq_shape)

    vertices = meshObstaculo1.vertices
    vertices3 = [(v[0], v[1]) for v in vertices]
    obstaculo1_der_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    obstaculo1_der_body.angle = np.pi/2
    obstaculo1_der_shape = pymunk.Poly(obstaculo1_der_body, vertices=vertices3)
    obstaculo1_der_body.position = (2, 0)

    world.add(obstaculo1_der_body, obstaculo1_der_shape)

    obstaculo1_izq_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    obstaculo1_izq_body.angle = 0
    obstaculo1_izq_shape = pymunk.Poly(obstaculo1_izq_body, vertices=vertices3)
    obstaculo1_izq_body.position = (-2, 0)

    world.add(obstaculo1_izq_body, obstaculo1_izq_shape)

    bodies = {}
    bodies['ball'] = ball_body
    bodies["flipper_der"] = flipper_der_body
    bodies["flipper_izq"] = flipper_izq_body


    #Se definen las vistas
    top_view = tr.lookAt(np.array([0, 0, 5]), np.array([0, 0, 0]), np.array([0, 1, 0]))
    pov =  tr.lookAt(np.array([0, -3, 3]), np.array([0, 0, 0]), np.array([0, 0, 1]))
    side_view = tr.lookAt(np.array([0, -.3, 0]), np.array([0, 0, 0]), np.array([0, 1, 1]))

    ortogonal = tr.ortho(-3, 3, -3, 3, 0.1, 100)
    perspective = tr.perspective(60, float(width) / float(height), 0.1, 100)

    TopView = "TopView"
    POV = "POV"
    SideView = "side_view"

    window.program_state = {
        "scene_graph": graph,
        "total_time": 0.0,
        "view": tr.lookAt(np.array([0, 0, 5]), np.array([0, 0, 0]), np.array([0, 1, 0])),
        "view_position": np.array([0, 0, 5]),
        "projection": ortogonal,
        "view_name": TopView,
        "flipper_der": {"angulo": 0,
                        "moving": False},
        "flipper_izq": {"angulo": 0,
                        "moving": False},
        }
    
    

    def ChangeView():
        if window.program_state["view_name"] == TopView:
            window.program_state["view"] = pov
            window.program_state["view_name"] = POV
            window.program_state["view_position"] = np.array([0, -5, 5])
            window.program_state["projection"] = perspective
        elif window.program_state["view_name"] == POV:
            window.program_state["view"] = side_view
            window.program_state["view_name"] = SideView
            window.program_state["view_position"] = np.array([0, -5, 0])
            window.program_state["projection"] = ortogonal
        elif window.program_state["view_name"] == SideView:
            window.program_state["view"] = top_view
            window.program_state["view_name"] = TopView
            window.program_state["view_position"] = np.array([0, 0, 5])
            window.program_state["projection"] = ortogonal             
        
    def MoveFlipperIzq():
        window.program_state["flipper_izq"]["moving"] = True  
        # flipper_izq_body.angular_velocity = 50
        # graph.nodes["flipper_izq_geometry"]["transform"] @= tr.rotationZ(flipper_der_body.angle)

    def ReturnFlipperIzq():
        window.program_state["flipper_izq"]["moving"] = False
        # flipper_izq_body.angular_velocity = 0
        # graph.nodes["flipper_izq_geometry"]["transform"] @= tr.rotationZ(-flipper_der_body.angle)

    def MoveFlipperDer():
        window.program_state["flipper_der"]["moving"] = True
        # bodies["flipper_der"].angle = window.program_state["flipper_der"]["angulo"]
        # flipper_der_body.angular_velocity = -50

    def ReturnFlipperDer():
        window.program_state["flipper_der"]["moving"] = False
        # flipper_der_body.angular_velocity = 0

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.C:
            ChangeView()
        if symbol == key.A:
            MoveFlipperIzq()
        if symbol == key.D:
            MoveFlipperDer()
        if symbol == key.SPACE:
            print(f"ball_position:\n{bodies["ball"].position}\n---------------\n")
            # print(f"flipper_position:\n{bodies["flipper_der"].position}\n---------------\n")
            print(f"flipper_angle:\n{bodies["flipper_der"].angle}\n---------------\n")
            flipper_der_body.angle = np.pi/4
            flipper_izq_body.angle = 3*np.pi/4
        if symbol == key.ENTER:
            ball_body.position = (0,0)
            ball_body.velocity = (0,0)
        if symbol == key.DOWN:
            bodies["ball"].position = (bodies["ball"].position[0], bodies["ball"].position[1] - 1)
            # ball_body.velocity = (0, -1)
        if symbol == key.UP:
            bodies["ball"].position = (bodies["ball"].position[0], bodies["ball"].position[1] + 1)
            # ball_body.velocity = (0, 1)
        if symbol == key.LEFT:
            bodies["ball"].position = (bodies["ball"].position[0] - .25, bodies["ball"].position[1])
            # ball_body.velocity = (-1, 0)

        if symbol == key.RIGHT:
            bodies["ball"].position = (bodies["ball"].position[0] + .25, bodies["ball"].position[1])
            # ball_body.velocity = (1, 0)
            

    @window.event
    def on_key_release(symbol, modifiers):
        if symbol == key.A:
            ReturnFlipperIzq()
        elif symbol == key.D:
             ReturnFlipperDer()
            

    graph = window.program_state["scene_graph"]
    root_key = graph.graph["root"]
    edges = list(nx.edge_dfs(graph, source=root_key))

    label = pyglet.text.Label('Presiona ESPACIO para desatascar flippers y ENTER para reiniciar el pelota',
                font_name='Times New Roman',
                font_size=15,
                color=(255,255,255,255),
                x= window.width // 2, y= 1.85 * window.height // 2,
                anchor_x='center', anchor_y='center')
    
    @window.event
    def on_draw():
        GL.glClearColor(0 , 0 , 0.04, 0.1)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)


        window.clear()
        tex_pipeline.use()

        tex_pipeline["view"] = window.program_state["view"].reshape(16, 1, order="F")
        tex_pipeline["projection"] = window.program_state["projection"].reshape(16, 1, order="F")
        tex_pipeline["view_position"] = window.program_state["view_position"]

        # flipper_izq_body.angle = window.program_state["flipper_izq"]["angulo"]
        # print(flipper_izq_body.angle)

        # if flipper_izq_body.angle >= np.pi/3:
        #     flipper_der_body.angular_velocity = 0

        if flipper_izq_body.angle -3*np.pi/4  >= np.pi/3:
            flipper_izq_body.angular_velocity = 0
        if flipper_izq_body.angle -3*np.pi/4  <= 0:
            flipper_izq_body.angular_velocity = 0

        if np.pi/2 - (flipper_der_body.angle + np.pi/4) >= np.pi/4:
            flipper_der_body.angular_velocity = 0
        if np.pi/2 - (flipper_der_body.angle + np.pi/4) <= 0:
            flipper_der_body.angular_velocity = 0
        # print(f"izq:{flipper_izq_body.angle -3*np.pi/4}\nder:{np.pi/2 - (flipper_der_body.angle + np.pi/4)}")
        
        # flipper_izq_body.angular_velocity = 0
        
        if window.program_state["flipper_izq"]["moving"] and (flipper_izq_body.angle -3*np.pi/4) < np.pi/3:
            flipper_izq_body.angular_velocity = 30

        if not window.program_state["flipper_izq"]["moving"] and (flipper_izq_body.angle -3*np.pi/4) > 0:
            flipper_izq_body.angular_velocity = -30
        
        # print(f"moving?: {window.program_state["flipper_izq"]["moving"]}, angulo: {flipper_izq_body.angle}")

        if window.program_state["flipper_der"]["moving"] and np.pi/2 - (flipper_der_body.angle + np.pi/4) < np.pi/3:
            flipper_der_body.angular_velocity = -30

        if not window.program_state["flipper_der"]["moving"] and np.pi/2 - (flipper_der_body.angle + np.pi/4) > 0:
            flipper_der_body.angular_velocity = 30

        transformations = {root_key: graph.nodes[root_key]["transform"]}

        for src, dst in edges:
            current_node = graph.nodes[dst]

            if not dst in transformations:
                dst_transform = current_node["transform"]
                transformations[dst] = transformations[src] @ dst_transform

            if "mesh" in current_node:

                if current_node["name"] not in ("ball_geometry", "flipper_izq_geometry", "flipper_der_geometry", "obstaculo1.2_geometry", "obstaculo1_geometry"):
                    tex_pipeline["transform"] = transformations[dst].reshape(16, 1, order="F")

                # if isinstance(current_node["mesh"], tm.Scene):
                #     vertex_lists = {}
                #     for object_id, object_geometry in current_node['mesh'].geometry.items():
                            
                #         mesh = {}
                #         object_geometry.fix_normals(True)
                #         object_vlist = tm.rendering.mesh_to_vertexlist(object_geometry)
                #         n_triangles = len(object_vlist[4][1]) // 3
                #         mesh["gpu_data"] = tex_pipeline.vertex_list_indexed(
                #             n_triangles, GL.GL_TRIANGLES, object_vlist[3]
                #         )
                #         mesh["gpu_data"].position[:] = object_vlist[4][1]
                #         mesh["gpu_data"].normal[:] = object_vlist[5][1]
                #         mesh["texture"] = texture_2D_setup(object_geometry.visual.material.image)
                #         mesh["gpu_data"].uv[:] = object_vlist[6][1]
                #         vertex_lists[object_id] = mesh

                #     if "texture" in vertex_lists[object_id]:
                #         # print(vertex_lists[object_id]["texture"])
                #         GL.glBindTexture(GL.GL_TEXTURE_2D, vertex_lists[object_id]["texture"])
                #     else:
                #         # esto "activa" una textura nula
                #         GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

                #     vertex_lists[object_id]["gpu_data"].draw(pyglet.gl.GL_TRIANGLES)
                    
                if isinstance(current_node["mesh"], pyglet.graphics.vertexdomain.IndexedVertexList):
                    for attr in current_node.keys():

                        for attr1 in current_node["mesh"].__dict__:
                            if attr1 in ("Kd", "Ks", "Ka"):
                                tex_pipeline[attr1] = current_node["mesh"].__dict__[attr1][:3] /255

                            elif attr1 == "ns":
                                tex_pipeline["ns"] = current_node["mesh"].__dict__[attr1]



                    if current_node["name"] == "ball_geometry":
                        x,y = bodies["ball"].position
                        alpha = bodies["ball"].angle
                        tex_pipeline["transform"] = (current_node["transform"] @ tr.translate(x, y, radio) @ tr.rotationZ(alpha)).reshape(16, 1, order="F")

                    elif current_node["name"] == "flipper_der_geometry":
                        x,y = flipper_der_body.position
                        alpha = flipper_der_body.angle
                        tex_pipeline["transform"] = (tr.translate(x/5, y/5, 0.125) @ current_node["transform"] @ tr.rotationZ(alpha)).reshape(16, 1, order="F")

                    elif current_node["name"] == "flipper_izq_geometry":
                        x,y = flipper_izq_body.position
                        alpha = flipper_izq_body.angle
                        tex_pipeline["transform"] = (tr.translate(x/5, y/5, 0.125) @ current_node["transform"] @ tr.rotationZ(alpha)).reshape(16, 1, order="F")
                    
                    elif current_node["name"] == "obstaculo1.2_geometry":
                        x,y = obstaculo1_der_body.position
                        alpha = obstaculo1_der_body.angle
                        tex_pipeline["transform"] = (tr.translate(x/5, y/5, 0.11) @ current_node["transform"] @ tr.rotationZ(alpha)).reshape(16, 1, order="F")

                    elif current_node["name"] == "obstaculo1_geometry":
                        x,y = obstaculo1_izq_body.position
                        alpha = obstaculo1_izq_body.angle
                        tex_pipeline["transform"] = (tr.translate(x/5, y/5, 0.11) @ current_node["transform"] @ tr.rotationZ(alpha)).reshape(16, 1, order="F")

                    GL.glBindTexture(GL.GL_TEXTURE_2D, current_node["mesh"].texture)
                    current_node["mesh"].draw(pyglet.gl.GL_TRIANGLES)
        
        label.draw()            

                
    pyglet.clock.schedule_interval(update_pinball, 1/60, window)
    pyglet.app.run()