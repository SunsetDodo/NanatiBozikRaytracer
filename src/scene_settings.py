class SceneSettings:

    # Hardcoded constants
    EPSILON = 1e-9
    SHADOW_RAY_RADIUS = 0.3

    def __init__(self, background_color, root_number_shadow_rays, max_recursions):
        self.background_color = background_color
        self.root_number_shadow_rays = root_number_shadow_rays
        self.max_recursions = max_recursions
