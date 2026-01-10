from __future__ import annotations

import numpy as np

from light import Light


class Material:
    def __init__(self, diffuse_color, specular_color, reflection_color, shininess, transparency):
        self.diffuse_color = np.array(diffuse_color)
        self.specular_color = np.array(specular_color)
        self.reflection_color = np.array(reflection_color)

        self.shininess = shininess
        self.transparency = transparency

    def calculate_light(self, light: Light,
                        normal_dir: np.array,
                        light_dir: np.array,
                        view_dir: np.array,
                        reflect_dir: np.array = None,
                        estimate: bool = False
                        ) -> np.array:

        n_dot_l = normal_dir @ light_dir
        if n_dot_l <= 0:
            return np.zeros(3, dtype=float)

        if estimate:
            highlight_dir = light_dir + view_dir
            highlight_dir = highlight_dir / np.linalg.norm(highlight_dir)
            v_dot_r = highlight_dir @ normal_dir
        else:
            if reflect_dir is None:
                reflect_dir = normal_dir * ((light_dir * 2) @ normal_dir) - light_dir
            v_dot_r = view_dir @ reflect_dir

        v_dot_r = max(0.0, v_dot_r)
        diffuse = light.color * self.diffuse_color * n_dot_l
        specular = light.color * self.specular_color * light.specular_intensity * pow(v_dot_r, self.shininess)

        return diffuse + specular
