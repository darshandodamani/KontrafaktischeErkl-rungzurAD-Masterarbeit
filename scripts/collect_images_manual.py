import carla
import random
import time
import os
import numpy as np
from PIL import Image
import pygame

# Initialize Pygame
os.environ["SDL_VIDEODRIVER"] = "x11"
pygame.init()
try:
    display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("CARLA Manual Control")
except pygame.error as e:
    print(f"Error initializing Pygame display: {e}")
    exit()

clock = pygame.time.Clock()

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Get the blueprint library and the spawn points
blueprint_library = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

# Choose a random vehicle blueprint and spawn point
vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
spawn_point = random.choice(spawn_points)

# Spawn the vehicle
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(False)

# Set up the camera sensor
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '110')

# Attach the camera to the vehicle
camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Create a directory to save the images
output_dir = 'carla_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define a callback function to save the images and display them
def save_and_display_image(image):
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # RGBA to RGB
    image_name = f"{output_dir}/image_{image.frame:06d}.png"
    Image.fromarray(array).save(image_name)
    print(f"Saved {image_name}")

    # Display the image
    try:
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        display.blit(surface, (0, 0))
        pygame.display.flip()
    except pygame.error as e:
        print(f"Pygame display error: {e}")

# Listen to the camera sensor
camera.listen(save_and_display_image)

# Function to control the vehicle manually
def control_vehicle(vehicle):
    control = carla.VehicleControl()
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        control.throttle = 1.0
    if keys[pygame.K_s]:
        control.brake = 1.0
    if keys[pygame.K_a]:
        control.steer = -1.0
    if keys[pygame.K_d]:
        control.steer = 1.0
    vehicle.apply_control(control)

try:
    while True:
        clock.tick(30)
        world.tick()
        control_vehicle(vehicle)

        # Capture Pygame events to allow manual control
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

finally:
    # Clean up
    camera.stop()
    vehicle.destroy()
    pygame.quit()
    print("All actors destroyed.")
