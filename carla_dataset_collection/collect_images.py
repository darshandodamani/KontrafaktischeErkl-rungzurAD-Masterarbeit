# collect_images.py
import carla
import random
import time
import os
import numpy as np
import cv2
import csv
import logging
import argparse

logging.basicConfig(level=logging.INFO)

# Utility Functions
def connect_to_carla():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    return client

def setup_world(client, town_name):
    logging.info(f"Loading {town_name}...")
    world = client.load_world(town_name)
    settings = world.get_settings()
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    logging.info("World loaded successfully.")
    return world

def setup_vehicle(world, vehicle_bp_name="vehicle.audi.a2"):
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find(vehicle_bp_name)
    spawn_points = world.get_map().get_spawn_points()
    start_point = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, start_point)
    vehicle.set_autopilot(True)
    logging.info("Vehicle spawned and autopilot enabled.")
    return vehicle

def setup_camera(world, vehicle, image_size_x=160, image_size_y=80, fov=125):
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(image_size_x))
    camera_bp.set_attribute("image_size_y", str(image_size_y))
    camera_bp.set_attribute("fov", str(fov))
    # camera_transform = carla.Transform(carla.Location(x=2.0, y=1.0, z=1.5))
    camera_transform = carla.Transform(carla.Location(x=2.0, y=0.0, z=1.5), carla.Rotation(pitch=-10))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    logging.info("Camera sensor attached to the vehicle.")
    return camera

# Dataset Collection Function
def collect_images(output_dir, town_name, image_size):
    client = connect_to_carla()
    world = setup_world(client, town_name)
    vehicle = setup_vehicle(world)
    camera = setup_camera(world, vehicle, image_size[0], image_size[1])
    
    image_array = None

    def process_image(image):
        nonlocal image_array
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        image_array = array[:, :, :3]

    camera.listen(process_image)
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', town_name + '_dataset')
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f"{town_name}_data_log.csv")

    try:
        frame = 0
        with open(csv_filename, "w", newline="") as csvfile:
            fieldnames = ["image_filename", "steering", "throttle", "brake"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            while frame < 20000:
                world.tick()
                time.sleep(0.2)

                if image_array is not None:
                    image_name = f"{town_name}_{frame:06d}.png"
                    image_path = os.path.join(output_dir, image_name)
                    cv2.imwrite(image_path, image_array)
                    logging.info(f"Saved {image_name}")

                    control = vehicle.get_control()
                    writer.writerow({
                        "image_filename": image_name,
                        "steering": control.steer,
                        "throttle": control.throttle,
                        "brake": control.brake,
                    })
                    frame += 1

    finally:
        if camera:
            camera.stop()
            camera.destroy()
        if vehicle:
            vehicle.destroy()
        logging.info("All actors destroyed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA Dataset Collection Script")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save collected dataset.")
    parser.add_argument("--town_name", type=str, default="Town03", help="CARLA town name to load.")
    parser.add_argument("--image_size", type=int, nargs=2, default=[160, 80], help="Image size (width height).")
    args = parser.parse_args()

    collect_images(args.output_dir, args.town_name, args.image_size)
