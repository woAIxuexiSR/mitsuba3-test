import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt

import time

from integrators import ambient, depth, normal, path, lhs, rhs, volpath
from integrators import test

mi.set_variant("cuda_ad_rgb")


def ball_cornell():

    scene_dict = mi.cornell_box()
    scene_dict.pop("small-box")
    scene_dict.pop("large-box")

    scene_dict["glass"] = {"type": "dielectric"}
    scene_dict["ball"] = {
        "type": "sphere",
        "to_world": mi.ScalarTransform4f.scale([0.4, 0.4, 0.4]).translate([0.5, 0, 0.5]),
        "bsdf": {"type": "ref", "id": "glass"}
    }

    return mi.load_dict(scene_dict)


def volume_box():

    sensor_mat = mi.ScalarTransform4f.rotate(
        [0, 1, 0], -90) @ mi.ScalarTransform4f.look_at(target=[0, 0, 0], origin=[0, 0, 4], up=[0, 1, 0])

    scene_dict = {
        'type': 'scene',
        'integrator': {'type': 'prbvolpath'},
        'object': {
            'type': 'cube',
            'bsdf': {'type': 'null'},
            'interior': {
                'type': 'heterogeneous',
                'sigma_t': {
                    'type': 'gridvolume',
                    'filename': './tutorials/scenes/volume.vol',
                    'to_world': mi.ScalarTransform4f.rotate([1, 0, 0], -90).scale(2).translate(-0.5)
                },
                'scale': 40
            }
        },
        'emitter': {'type': 'constant'},
        'sensor': {
            'type': 'perspective',
            'fov': 45,
            'to_world': sensor_mat,
            'film': {
                'type': 'hdrfilm',
                'width': 256, 'height': 256,
                'filter': {'type': 'tent'}
            }
        }
    }

    return mi.load_dict(scene_dict)


if __name__ == "__main__":

    # scene_dict = mi.cornell_box()
    # scene = mi.load_dict(scene_dict)

    # scene = mi.load_file("./scenes/cornell-box/scene.xml")
    # scene = mi.load_file("./scenes/dining-room/scene.xml")
    # scene = mi.load_file("./scenes/living-room/scene.xml")
    # scene = mi.load_file("./scenes/living-room-3/scene.xml")
    # scene = mi.load_file("./scenes/veach-ajar/scene.xml")
    # scene = mi.load_file("./scenes/veach-bidir/scene.xml")
    scene = mi.load_file("./scenes/volumetric-caustic/scene.xml")
    # scene = mi.load_file("./scenes/banner_06/scene.xml")
    # scene = ball_cornell()
    # scene = volume_box()


    start = time.time()

    integrator = mi.load_dict({
        "type": "volpath",
        "max_depth": 16,
        # "rr_depth": 3,
    })

    # img = mi.Bitmap(mi.render(scene, integrator=integrator, spp=16))

    spp = 128

    sensor: mi.Sensor = scene.sensors()[0]
    film: mi.Film = sensor.film()
    width, height = film.crop_size()

    img = mi.TensorXf(dr.zeros(float, (height, width, 3)))
    for i in range(spp):
        img += mi.render(scene, integrator=integrator, spp=1, seed=i)
    img /= spp
    img: mi.Bitmap = mi.Bitmap(img)
    img.write("ha3.exr")

    end = time.time()
    print(f"Time: {end - start}")

    img = img.convert(component_format=mi.Bitmap.UInt8, srgb_gamma=True)
    plt.imshow(img)
    plt.show()
