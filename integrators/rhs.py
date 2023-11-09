import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")


def first_smooth(
    scene: mi.Scene,
    sampler: mi.Sampler,
    ray: mi.Ray3f,
    active: bool = True
) -> tuple[mi.SurfaceInteraction3f, mi.Color3f, bool]:

    ray = mi.Ray3f(ray)
    si: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
    active = mi.Bool(active)
    throughput = mi.Color3f(1.0)
    depth = mi.UInt32(0)
    null_face = mi.Bool(True)

    bsdf_ctx = mi.BSDFContext()

    loop = mi.Loop(
        "first smooth surface",
        lambda: (
            sampler,
            ray,
            si,
            active,
            throughput,
            depth,
            null_face,
        )
    )

    max_depth = 5
    loop.set_max_iterations(max_depth)

    while loop(active):

        si = scene.ray_intersect(ray, active)
        bsdf: mi.BSDF = si.bsdf(ray)

        null_face &= mi.has_flag(bsdf.flags(), mi.BSDFFlags.Null) | (
            ~mi.has_flag(bsdf.flags(), mi.BSDFFlags.BackSide) & (
                si.wi.z < 0)
        )

        active &= si.is_valid() & ~null_face & (depth < max_depth)
        active &= ~mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

        bsdf_sample, bsdf_weight = bsdf.sample(
            bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active
        )
        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
        throughput[active] *= bsdf_weight
        depth[si.is_valid()] += 1

    return si, throughput, null_face


class RHSIntegrator(mi.SamplingIntegrator):
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
        result = mi.Color3f(0.0)

        si, throughput, null_face = first_smooth(scene, sampler, ray, active)
        bsdf: mi.BSDF = si.bsdf()
        bsdf_ctx = mi.BSDFContext()

        active_next = si.is_valid() & ~null_face
        result[active_next] += throughput * si.emitter(scene).eval(si)

        # emitter sampling
        active_em = active_next & mi.has_flag(
            bsdf.flags(), mi.BSDFFlags.Smooth)

        ds, em_weight = scene.sample_emitter_direction(
            si, sampler.next_2d(), True, active_em
        )
        active_em &= dr.neq(ds.pdf, 0.0)

        wo = si.to_local(ds.d)
        bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(
            bsdf_ctx, si, wo, active_em
        )
        mis_em = dr.select(ds.delta, 1.0, dr.detach(
            dr.select(ds.pdf > 0, ds.pdf / ds.pdf + bsdf_pdf_em, 0)
        ))
        result[active_em] += throughput * (em_weight * bsdf_value_em * mis_em)

        # bsdf sampling
        bsdf_sample, bsdf_weight = bsdf.sample(
            bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_next
        )
        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
        throughput *= bsdf_weight

        prev_si: mi.SurfaceInteraction3f = dr.detach(si, True)
        prev_bsdf_pdf: mi.Float = bsdf_sample.pdf
        prev_bsdf_delta: mi.Bool = mi.has_flag(
            bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

        si = scene.ray_intersect(ray)

        ds: mi.DirectionSample3f = mi.DirectionSample3f(scene, si, prev_si)
        em_pdf = mi.Float(0.0)
        em_pdf = scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
        mis_weight = dr.select(
            prev_bsdf_pdf > 0, prev_bsdf_pdf / (prev_bsdf_pdf + em_pdf), 0)

        # next smooth surface
        si, throughput2, null_face = first_smooth(
            scene, sampler, ray, active_next)

        active = si.is_valid() & ~null_face & active_next
        result[active] += throughput * throughput2 * mis_weight * (
            si.bsdf().eval_diffuse_reflectance(si) + si.emitter(scene).eval(si)
        )

        return result, active, []


mi.register_integrator("rhs", lambda props: RHSIntegrator(props))
