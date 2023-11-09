import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")


class TestIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.max_depth = props.get("max_depth", 16)
        self.rr_depth = props.get("rr_depth", 3)

    def mis_weight(self, p1, p2):
        return dr.detach(dr.select(p1 > 0.0, p1 / (p1 + p2), 0.0))

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
        result = mi.Color3f(0.0)
        depth = mi.UInt32(0)

        valid_ray = mi.Bool(scene.environment() is not None)

        si: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
        mei: mi.MediumInteraction3f = dr.zeros(mi.MediumInteraction3f)
        medium: mi.MediumPtr = medium if medium is not None else dr.zeros(
            mi.MediumPtr)

        # Variables caching information from the previous bounce
        needs_intersection = mi.Bool(True)
        specular_chain = mi.Bool(active)
        last_scatter_event: mi.Interaction3f = dr.zeros(mi.Interaction3f)
        last_scatter_direction_pdf = mi.Float(1.0)

        # Variables unchanged during the loop
        channel = mi.UInt32(0)
        channel = dr.minimum(sampler.next_1d(active) * 3, 2)
        bsdf_ctx: mi.BSDFContext = mi.BSDFContext()
        phase_ctx: mi.PhaseFunctionContext = mi.PhaseFunctionContext(
            sampler)

        loop = mi.Loop("Volpath integrator", lambda: (
            active, depth, ray, throughput, result, si, mei, medium,
            last_scatter_event, last_scatter_direction_pdf,
            needs_intersection, specular_chain, valid_ray, sampler
        ))

        loop.set_max_iterations(self.max_depth)

        while loop(active):

            # ---------------------- Surface Interactions ----------------------

            active_surface = mi.Bool(active)
            intersect = active_surface & needs_intersection
            si[intersect] = scene.ray_intersect(ray, intersect)

            # intersection with emitters
            emitter: mi.Emitter = si.emitter(scene)
            active_e = active_surface & dr.neq(emitter, None)
            count_direct = (active_surface & dr.eq(depth, 0)) | specular_chain

            emitter_pdf = mi.Float(1.0)
            ds: mi.DirectionSample3f = mi.DirectionSample3f(
                scene, si, last_scatter_event)
            emitter_pdf = scene.pdf_emitter_direction(
                last_scatter_event, ds, active_e)
            emitted = emitter.eval(si, active_e)

            result[active_e & count_direct] += throughput * emitted
            result[active_e & ~count_direct] += throughput * emitted * self.mis_weight(
                last_scatter_direction_pdf, emitter_pdf)

            # next event estimation
            active_surface &= si.is_valid()
            bsdf: mi.BSDF = si.bsdf(ray)
            active_e = active_surface & mi.has_flag(
                bsdf.flags(), mi.BSDFFlags.Smooth)

            ds, emitted = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_e
            )

            wo = si.to_local(ds.d)

            sample1 = sampler.next_1d()
            sample2 = sampler.next_2d()

            bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight = bsdf.eval_pdf_sample(
                bsdf_ctx, si, wo, sample1, sample2
            )

            bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)
            result[active_e] += throughput * bsdf_val * self.mis_weight(
                ds.pdf, dr.select(ds.delta, 0, bsdf_pdf)) * emitted

            # BSDF sampling
            bsdf_weight = si.to_world_mueller(
                bsdf_weight, -bsdf_sample.wo, si.wi)

            ray[active_surface] = si.spawn_ray(si.to_world(bsdf_sample.wo))
            throughput[active_surface] *= bsdf_weight

            # update informations
            needs_intersection |= active_surface

            non_null_bsdf = active_surface & ~mi.has_flag(
                bsdf_sample.sampled_type, mi.BSDFFlags.Null)
            depth[non_null_bsdf] += 1

            last_scatter_event[non_null_bsdf] = si
            last_scatter_direction_pdf[non_null_bsdf] = bsdf_sample.pdf

            valid_ray |= non_null_bsdf
            specular_chain |= non_null_bsdf & mi.has_flag(
                bsdf_sample.sampled_type, mi.BSDFFlags.Delta)
            specular_chain &= ~(active_surface & mi.has_flag(
                bsdf_sample.sampled_type, mi.BSDFFlags.Smooth))

            medium[active_surface & si.is_medium_transition()
                   ] = si.target_medium(ray.d)

            # ---------------------- Russian Roulette ----------------------

            throughput_max = dr.max(mi.unpolarized_spectrum(throughput))

            rr_prop = dr.minimum(throughput_max, 0.95)
            rr_active = (depth >= self.rr_depth)
            rr_continue = (sampler.next_1d() < rr_prop)

            throughput[rr_active] *= dr.rcp(rr_prop)

            active = (
                active & (active_surface) & (
                    ~rr_active | rr_continue) & (dr.neq(throughput_max, 0.0))
            )
            active &= (depth < self.max_depth)

        return result, valid_ray, []


mi.register_integrator("test", lambda props: TestIntegrator(props))
