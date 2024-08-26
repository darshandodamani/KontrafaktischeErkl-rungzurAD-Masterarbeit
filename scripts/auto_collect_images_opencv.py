import carla
import random
import time
import os
import numpy as np
import cv2

# Directory to save images
output_dir = 'dataset/town7_dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

    # Choose a random start point
    start_point = random.choice(spawn_points)  # Random start point

    # Spawn the vehicle at the start point
    vehicle_bp = blueprint_library.find('vehicle.audi.a2')
    vehicle = world.spawn_actor(vehicle_bp, start_point)
    vehicle.set_autopilot(True)  # Enable autopilot

    # Set up the camera sensor
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '160')
    camera_bp.set_attribute('image_size_y', '80')
    camera_bp.set_attribute('fov', '125')

    # Attach the camera to the vehicle at the desired position
    camera_transform = carla.Transform(carla.Location(x=2.0, y=1.0, z=1.5), carla.Rotation(pitch=0, yaw=0, roll=0))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Global variable to store the image
    image_array = None

    # Define a callback function to save the images
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

        # Continuous data capture loop
        while True:
            world.tick()
            time.sleep(0.2)  # Capture images more frequently (adjust as needed)

            if image_array is not None:
                image_name = os.path.join(output_dir, f"town7_{frame:06d}.png")
                cv2.imwrite(image_name, image_array)
                print(f"Saved {image_name}")
                frame += 1

    finally:
        # Clean up
        camera.stop()
        vehicle.destroy()
        cv2.destroyAllWindows()
        print("All actors destroyed.")

except RuntimeError as e:
    print(f"Error: {e}")
