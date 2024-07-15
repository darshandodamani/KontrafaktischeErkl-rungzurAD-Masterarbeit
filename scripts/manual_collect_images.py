import carla
import random
import time
import os
import numpy as np
import pygame

# Directory to save images
save_dir = "collected_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("CARLA Manual Control")

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(30.0)

try:
    world = client.load_world('Town03')  # or any other town you prefer
    settings = world.get_settings()
    settings.no_rendering_mode = True
    world.apply_settings(settings)

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

    frame = 0

    # Define a callback function to save the images
    def save_image(image):
        global frame
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # RGBA to RGB
        image_filename = os.path.join(save_dir, f"image_{frame:06d}.png")
        carla.Image.save_to_disk(image, image_filename)
        frame += 1
        print(f"Saved image {image_filename}")

    # Listen to the camera sensor
    camera.listen(save_image)

    # Run the loop to keep capturing images
    try:
        while True:
            world.tick()
            time.sleep(0.1)  # Adjust as needed to control the rate of image capture

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        save_image_flag = True

            keys = pygame.key.get_pressed()
            control = carla.VehicleControl()
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                control.throttle = 1.0
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                control.brake = 1.0
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                control.steer = -1.0
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                control.steer = 1.0
            vehicle.apply_control(control)

    finally:
        # Clean up
        camera.stop()
        vehicle.destroy()
        pygame.quit()
        print("All actors destroyed.")

except RuntimeError as e:
    print(f"Error: {e}")
