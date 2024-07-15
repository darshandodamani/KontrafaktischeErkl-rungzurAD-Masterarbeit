import carla
import random
import time
import os
import numpy as np
import cv2

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

    # Create a global variable to store the image array
    global image_array
    image_array = None

    # Define a callback function to display the image using OpenCV
    def display_image(image):
        global image_array
        try:
            image.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            array = array[:, :, :3]  # RGBA to RGB
            array = array[:, :, ::-1]  # RGB to BGR for OpenCV

            # Confirm image data is received
            print(f"Received image data: {array.shape}")
            print(f"Image data (first 5 pixels): {array[0, :5, :]}")

            image_array = array
        except Exception as e:
            print(f"Error processing image: {e}")

    # Listen to the camera sensor
    camera.listen(display_image)

    # Run the loop to keep the window open and display the camera feed
    try:
        while True:
            if image_array is not None:
                cv2.imshow("CARLA Camera", image_array)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            world.tick()
            time.sleep(0.03)  # Add a small delay to prevent CPU overuse

    finally:
        # Clean up
        camera.stop()
        vehicle.destroy()
        cv2.destroyAllWindows()
        print("All actors destroyed.")

except RuntimeError as e:
    print(f"Error: {e}")
