import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")


class LHSIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.max_depth = props.get("max_depth", 16)

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, bool, list[float]]:

        ray = mi.Ray3f(ray)
        active = mi.Bool(active)
        throughput = mi.Color3f(1.0)
        depth = mi.UInt32(0)
        null_face = mi.Bool(True)

        bsdf_ctx = mi.BSDFContext()

        loop = mi.Loop(
            "left-hand side",
            lambda: (
                sampler,
                ray,
                active,
                throughput,
                depth,
                null_face,
            )
        )

        loop.set_max_iterations(self.max_depth)

        while loop(active):

            si: mi.SurfaceInteraction3f = scene.ray_intersect(ray, active)
            bsdf: mi.BSDF = si.bsdf(ray)

            null_face &= mi.has_flag(bsdf.flags(), mi.BSDFFlags.Null) | (
                ~mi.has_flag(bsdf.flags(), mi.BSDFFlags.BackSide) & (
                    si.wi.z < 0)
            )

            active &= si.is_valid() & ~null_face & (depth < self.max_depth)
            active &= ~mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            bsdf_sample, bsdf_weight = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active
            )
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            throughput[active] *= bsdf_weight
            depth[si.is_valid()] += 1

        result = mi.Color3f(0.0)
        result[~null_face] = throughput * (si.bsdf().eval_diffuse_reflectance(si)
                                           + si.emitter(scene).eval(si))

        return result, ~null_face, []


mi.register_integrator("lhs", lambda props: LHSIntegrator(props))
