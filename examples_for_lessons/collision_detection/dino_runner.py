import pyglet
from OpenGL.GL import *

import sys, os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import grafica.easy_shaders as es

from models import Player
from controller import Controller

from models import game_state, ObstacleManager

"""
The idea of this project is to something way more complex. Since this is an advanced lesson,
students should be able to understand it better
"""



if __name__ == '__main__':
    # We add this import here only to make it more explicit
    # note that we are importing a variable instead of a class
    game_state.player = Player(x=-0.8)
    game_state.obstacle_manager = ObstacleManager()
    game_state.obstacle_manager.generate_obstacles(3)

    # Create Window
    controller: Controller = Controller(width=1280, height=800, game_state=game_state)
    controller.current_pipeline = es.SimpleTextureTransformShaderProgram()
    
    game_state.player.set_gpu_shape(controller.current_pipeline)
    game_state.set_gpu_shapes_of_obstacles(controller.current_pipeline)

    pyglet.clock.schedule(game_state.update)
    pyglet.clock.schedule(game_state.player.update, controller)
    pyglet.app.run()