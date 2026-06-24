class SceneSettings:

    def __init__(self, background_color, root_number_shadow_rays, max_bounce_depth: int = 5):
        self.background_color = background_color
        self.root_number_shadow_rays = root_number_shadow_rays
        self.max_bounce_depth = int(max_bounce_depth)
