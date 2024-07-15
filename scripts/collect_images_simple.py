import carla
import random
import time
import os
import numpy as np
import pygame

# Initialize Pygame
os.environ["SDL_VIDEODRIVER"] = "x11"
pygame.init()
try:
    display = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("CARLA Simple Test")
    print("Pygame display initialized.")
except pygame.error as e:
    print(f"Error initializing Pygame display: {e}")
    exit()

clock = pygame.time.Clock()

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(30.0)

try:
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

    # Confirm camera initialization
    print("Camera initialized and attached to the vehicle.")

    # Define a callback function to display the image
    def display_image(image):
        try:
            image.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            array = array[:, :, :3]  # RGBA to RGB

            # Confirm image data is received
            print(f"Received image data: {array.shape}")

            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))
            pygame.display.flip()
            print("Image displayed.")
        except Exception as e:
            print(f"Error processing image: {e}")

    # Listen to the camera sensor
    camera.listen(display_image)

    # Run the loop to keep the window open and display the camera feed
    try:
        while True:
            clock.tick(30)
            world.tick()

            # Capture Pygame events to keep the window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

    finally:
        # Clean up
        camera.stop()
        vehicle.destroy()
        pygame.quit()
        print("All actors destroyed.")

except RuntimeError as e:
    print(f"Error: {e}")
