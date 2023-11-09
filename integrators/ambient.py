import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")

class AmbientIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties):
        super().__init__(props)

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, bool, list[float]]:

        # Render ambient

        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)

        bsdf: mi.BSDF = si.bsdf(ray)
        color = bsdf.eval_diffuse_reflectance(si)

        result = mi.Color3f(0.0)
        result[si.is_valid()] = color * (0.2 + 0.8 * abs(dr.dot(si.n, -ray.d)))

        return result, si.is_valid(), []

mi.register_integrator("ambient", lambda props: AmbientIntegrator(props))
