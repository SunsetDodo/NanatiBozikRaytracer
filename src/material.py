from __future__ import annotations

from light import Light
from vector3 import Vector3, dot, vec3_convolution


class Material:
    def __init__(self, diffuse_color, specular_color, reflection_color, shininess, transparency):
        self.diffuse_color = Vector3.from_array(diffuse_color)
        self.specular_color = Vector3.from_array(specular_color)
        self.reflection_color = Vector3.from_array(reflection_color)

        self.shininess = shininess
        self.transparency = transparency

    def calculate_light(self, light: Light,
                        normal_dir: Vector3,
                        light_dir: Vector3,
                        view_dir: Vector3,
                        reflect_dir: Vector3 = None,
                        estimate: bool = False
                        ) -> Vector3:

        if estimate:
            highlight_dir = light_dir + view_dir
            v_dot_r = dot(highlight_dir, normal_dir)
        else:
            if reflect_dir is None:
                reflect_dir = (normal_dir * dot(light_dir * 2, normal_dir)) - light_dir
            v_dot_r = dot(view_dir, reflect_dir)

        n_dot_l = dot(normal_dir, light_dir)

        diffuse = vec3_convolution(light.color, self.diffuse_color) * n_dot_l
        specular = vec3_convolution(light.color, self.specular_color) * light.specular_intensity * pow(v_dot_r,
                                                                                                       self.shininess)

        return diffuse + specular
