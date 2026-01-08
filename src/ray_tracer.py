import argparse
from PIL import Image
import numpy as np
import os
import sys
import datetime
import logging
import tqdm

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from ray import Ray, trace_ray
from viewport import Viewport

from scene import Scene



def setup_logger(log_level = logging.INFO):
    os.makedirs("logs", exist_ok=True)
    logfile = os.path.join("logs", f"{datetime.datetime.now().isoformat()}.log")
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logfile, encoding="utf8")
        ]
    )


def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    s = Scene()

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
                s.camera = camera
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
                s.settings = scene_settings
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
                s.materials.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
                s.surfaces.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
                s.surfaces.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
                s.surfaces.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
                s.lights.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return s


def save_image(image_array, path):
    image = Image.fromarray(np.uint8(image_array * 255))
    # Save the image to a file
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=600, help='Image width')
    parser.add_argument('--height', type=int, default=400, help='Image height')
    args = parser.parse_args()
    setup_logger(logging.DEBUG)
    logger = logging.getLogger("Raytracer").getChild("Main")

    # TODO - maybe remove me
    aspect_ratio = args.width / args.height
    logger.info("Starting Raytracing, width: %d height: %d (Aspect Ratio is: %.2f)", args.width, args.height, aspect_ratio)

    # Parse the scene file
    s = parse_scene_file(args.scene_file)
    image_array = np.zeros((args.height, args.width, 3))

    vp = Viewport(s.camera, args.width, args.height)
    origin = s.camera.position
    for x in tqdm.tqdm(range(args.width), desc="Rendering"):
        for y in range(args.height):
            target = vp.get_pixel_center(x, y)
            r = Ray(origin, target - origin)
            color = np.clip(trace_ray(s, r, s.settings.max_recursions), 0.0, 1.0)
            image_array[y][x] = color

    save_image(image_array, args.output_image)


if __name__ == '__main__':
    main()
