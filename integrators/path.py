import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")


class PathTracer(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.max_depth = props.get("max_depth", 16)
        self.rr_depth = props.get("rr_depth", 3)
        self.use_nee = props.get("use_nee", True)
        self.use_mis = props.get("use_mis", True)

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

        # Variables caching information from the previous bounce
        prev_si: mi.Interaction3f = dr.zeros(mi.Interaction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
        bsdf_ctx = mi.BSDFContext()

        loop = mi.Loop(
            "Path Tracer",
            lambda: (
                sampler,
                ray,
                throughput,
                result,
                depth,
                valid_ray,
                prev_si,
                prev_bsdf_pdf,
                prev_bsdf_delta,
                active,
            )
        )

        loop.set_max_iterations(self.max_depth)

        while loop(active):

            si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)

            # ---------------------- Direct emission ----------------------

            mis_weight = mi.Float(1.0)

            if self.use_nee:

                mis_weight[~prev_bsdf_delta] = 0.0

                if self.use_mis:

                    ds: mi.DirectionSample3f = mi.DirectionSample3f(
                        scene, si, prev_si)
                    em_pdf = mi.Float(0.0)

                    em_pdf = scene.pdf_emitter_direction(
                        prev_si, ds, ~prev_bsdf_delta)

                    mis_bsdf = dr.detach(
                        dr.select(prev_bsdf_pdf > 0, prev_bsdf_pdf / (prev_bsdf_pdf + em_pdf), 0))
                    mis_weight[~prev_bsdf_delta] = mis_bsdf

            result = dr.fma(
                throughput,
                si.emitter(scene).eval(si) * mis_weight,
                result
            )

            active_next = ((depth + 1) < self.max_depth) & si.is_valid()

            bsdf: mi.BSDF = si.bsdf(ray)

            # ---------------------- Emitter sampling ----------------------

            if self.use_nee:

                active_em = active_next & mi.has_flag(
                    bsdf.flags(), mi.BSDFFlags.Smooth)

                ds, em_weight = scene.sample_emitter_direction(
                    si, sampler.next_2d(), True, active_em
                )
                active_em &= dr.neq(ds.pdf, 0.0)

                wo = si.to_local(ds.d)

            # ------ Evaluate BSDF * cos(theta) and sample direction -------

                sample1 = sampler.next_1d()
                sample2 = sampler.next_2d()

                bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight = bsdf.eval_pdf_sample(
                    bsdf_ctx, si, wo, sample1, sample2
                )

            # --------------- Emitter sampling contribution ----------------

                bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)

                mi_em = mi.Float(1.0)
                if self.use_mis:
                    mi_em = dr.select(ds.delta, 1.0, dr.detach(
                        dr.select(ds.pdf > 0, ds.pdf / (ds.pdf + bsdf_pdf), 0)))

                result[active_em] = dr.fma(
                    throughput, bsdf_val * em_weight * mi_em, result)

            else:

                sample1 = sampler.next_1d()
                sample2 = sampler.next_2d()

                bsdf_sample, bsdf_weight = bsdf.sample(
                    bsdf_ctx, si, sample1, sample2)

            # ---------------------- BSDF sampling ----------------------

            bsdf_weight = si.to_world_mueller(
                bsdf_weight, -bsdf_sample.wo, si.wi)

            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

            # ------ Update loop variables based on current interaction ------

            throughput *= bsdf_weight
            valid_ray |= (
                active
                & si.is_valid()
                & ~mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Null)
            )

            prev_si = mi.Interaction3f(si)
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(
                bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # -------------------- Stopping criterion ---------------------

            depth[si.is_valid()] += 1

            throughput_max = dr.max(throughput)

            rr_prop = dr.minimum(throughput_max, 0.95)
            rr_active = (depth >= self.rr_depth)
            rr_continue = (sampler.next_1d() < rr_prop)

            throughput[rr_active] *= dr.rcp(rr_prop)

            active = (
                active_next & (~rr_active | rr_continue) & (
                    dr.neq(throughput_max, 0.0))
            )

        return dr.select(valid_ray, result, 0.0), valid_ray, []


mi.register_integrator("path_tracer", lambda props: PathTracer(props))
