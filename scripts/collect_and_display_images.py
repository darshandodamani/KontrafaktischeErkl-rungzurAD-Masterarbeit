import carla
import random
import time
import os
import numpy as np
import cv2

# Directory to save images
save_dir = "collected_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

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
    vehicle.set_autopilot(True)

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
    def save_and_display_image(image):
        global frame
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # RGBA to RGB
        array = array[:, :, ::-1]  # RGB to BGR for OpenCV

        image_filename = os.path.join(save_dir, f"image_{frame:06d}.png")
        cv2.imwrite(image_filename, array)
        print(f"Saved image {image_filename}")

        # Display the image
        cv2.imshow("CARLA Camera", array)
        cv2.waitKey(1)

        frame += 1

    # Listen to the camera sensor
    camera.listen(save_and_display_image)

    # Run the loop to keep capturing images
    try:
        while True:
            world.tick()
            time.sleep(0.1)  # Adjust as needed to control the rate of image capture

    finally:
        # Clean up
        camera.stop()
        vehicle.destroy()
        cv2.destroyAllWindows()
        print("All actors destroyed.")

except RuntimeError as e:
    print(f"Error: {e}")
