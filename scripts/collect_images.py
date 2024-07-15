import carla
import random
import time
import os
import numpy as np
import pygame
import cv2
from PIL import Image

# Initialize Pygame for key event handling
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("CARLA Manual Control")

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

try:
    # Load Town 7
    world = client.load_world('Town07')
    settings = world.get_settings()
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    # Get the blueprint library and the spawn points
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # Choose a specific vehicle blueprint (Audi vehicle)
    vehicle_bp = blueprint_library.find('vehicle.audi.a2')  # Example: Audi A2
    spawn_point = random.choice(spawn_points)

    # Spawn the vehicle
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(False)

    # Set up the camera sensor
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '110')

    # Attach the camera to the vehicle at the desired position
    # Example: Camera placed at the front right corner of the vehicle
    camera_transform = carla.Transform(carla.Location(x=2.0, y=1.0, z=1.5), carla.Rotation(pitch=0, yaw=0, roll=0))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Create a directory to save the images
    output_dir = 'carla_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Global variable to store the image
    image_array = None

    # Define a callback function to save the images and display them
    def process_image(image):
        global image_array
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        image_array = array

    # Listen to the camera sensor
    camera.listen(process_image)

    try:
        frame = 0
        while True:
            world.tick()
            time.sleep(0.03)  # Adjust as needed to control the rate of image capture

            if image_array is not None:
                cv2.imshow("CARLA Camera", image_array)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    image_name = os.path.join(output_dir, f"image_{frame:06d}.png")
                    cv2.imwrite(image_name, image_array)
                    print(f"Saved {image_name}")
                    frame += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            keys = pygame.key.get_pressed()
            control = carla.VehicleControl()
            
            # Vehicle control mappings
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
        cv2.destroyAllWindows()
        pygame.quit()
        print("All actors destroyed.")

except RuntimeError as e:
    print(f"Error: {e}")
