import carla
import random
import time
import os
import numpy as np
import cv2
import csv

# Directory to save images
output_dir = "dataset/town10_dataset"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# CSV file to store image filenames and control values
csv_filename = os.path.join(output_dir, "town10_data_log.csv")

# Connect to the CARLA server
print("Connecting to CARLA server...")
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
print(world.get_map().name)

try:
    # Load Town 7
    print("Loading Town 10...")
    world = client.load_world("Town10HD")
    print(world.get_map().name)
    settings = world.get_settings()
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    print("World loaded. Spawning vehicle...")

    # Get the blueprint library and the spawn points
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # Choose a random start point
    start_point = random.choice(spawn_points)  # Random start point

    # Spawn the vehicle at the start point
    vehicle_bp = blueprint_library.find("vehicle.audi.a2")
    vehicle = world.spawn_actor(vehicle_bp, start_point)
    vehicle.set_autopilot(True)  # Enable autopilot

    # Set up the camera sensor
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "256")
    camera_bp.set_attribute("image_size_y", "256")
    camera_bp.set_attribute("fov", "125")

    # Attach the camera to the vehicle at the desired position
    camera_transform = carla.Transform(
        carla.Location(x=2.0, y=1.0, z=1.5), carla.Rotation(pitch=0, yaw=0, roll=0)
    )
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

    # Open CSV file for writing
    with open(csv_filename, "w", newline="") as csvfile:
        fieldnames = ["image_filename", "steering", "throttle", "brake"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        try:
            frame = 0

            # Continuous data capture loop
            while True:
                world.tick()
                time.sleep(0.2)  # Capture images more frequently ( as needed)

                if image_array is not None:
                    image_name = f"town10_{frame:06d}.png"
                    image_path = os.path.join(output_dir, image_name)
                    cv2.imwrite(image_path, image_array)
                    print(f"Saved {image_name}")

                    # Get the current vehicle control values
                    control = vehicle.get_control()
                    steering = control.steer
                    throttle = control.throttle
                    brake = control.brake

                    # Save the data to the CSV file
                    writer.writerow(
                        {
                            "image_filename": image_name,
                            "steering": steering,
                            "throttle": throttle,
                            "brake": brake,
                        }
                    )

                    frame += 1

        finally:
            # Clean up
            camera.stop()
            vehicle.destroy()
            cv2.destroyAllWindows()
            print("All actors destroyed.")

except RuntimeError as e:
    print(f"Error: {e}")
