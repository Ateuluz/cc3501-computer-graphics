import pymunk
import pymunk.pygame_util
import pygame

def create_flipper(space, position, size):
    # Create the kinematic flipper body
    flipper_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    flipper_body.position = position

    # Define the shape of the flipper
    flipper_shape = pymunk.Poly.create_box(flipper_body, size)
    flipper_shape.density = 1
    space.add(flipper_body, flipper_shape)

    # Create a static body for the pivot point
    static_body = space.static_body

    # Create a pivot joint to anchor the flipper to the static body
    pivot_joint = pymunk.PivotJoint(static_body, flipper_body, position)
    space.add(pivot_joint)

    return flipper_body

# Initialize Pygame and Pymunk
pygame.init()
screen = pygame.display.set_mode((600, 600))
clock = pygame.time.Clock()
space = pymunk.Space()

# Create a flipper
position = (300, 300)  # Pivot point
size = (100, 20)       # Flipper size (width, height)
flipper_body = create_flipper(space, position, size)

# Create the Pygame/Pymunk draw options
draw_options = pymunk.pygame_util.DrawOptions(screen)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                flipper_body.angular_velocity = 10  # Set angular velocity when space is pressed
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                flipper_body.angular_velocity = -10  # Reverse angular velocity when space is released

    # Clear screen
    screen.fill((255, 255, 255))

    # Step the physics
    space.step(1 / 60.0)

    # Draw the objects
    space.debug_draw(draw_options)

    # Flip the screen
    pygame.display.flip()

    # Delay to control frame rate
    clock.tick(60)

pygame.quit()
