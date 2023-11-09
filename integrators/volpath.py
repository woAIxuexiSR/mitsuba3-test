import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")


class VolPathTracer(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.max_depth = props.get("max_depth", 16)
        self.rr_depth = props.get("rr_depth", 3)

    def index_spectrum(self, spec, idx):
        m = spec[0]
        m[dr.eq(idx, 1)] = spec[1]
        m[dr.eq(idx, 2)] = spec[2]
        return m

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

            # ---------------------- Sampling the RTE ----------------------

            active_medium = active & dr.neq(medium, None)
            active_surface = active & ~active_medium

            act_null_scatter = mi.Bool(False)
            act_medium_scatter = mi.Bool(False)
            escaped_medium = mi.Bool(False)

            # sample distance and compare to the maximum distance
            mei = medium.sample_interaction(ray, sampler.next_1d(
                active_medium), channel, active_medium)

            ray.maxt[active_medium & mei.is_valid()] = mei.t
            # intersect = needs_intersection & active_medium
            # si[intersect] = scene.ray_intersect(ray, intersect)
            si[active_medium] = scene.ray_intersect(ray, active_medium)

            # update the throughput for the free flight
            mei.t[active_medium & (si.t < mei.t)] = dr.inf
            tr, free_flight_pdf = medium.transmittance_eval_pdf(
                mei, si, active_medium)
            tr_pdf = self.index_spectrum(free_flight_pdf, channel)
            throughput[active_medium] *= dr.select(
                tr_pdf > 0.0, tr / tr_pdf, 0.0)

            needs_intersection &= ~active_medium
            escaped_medium = active_medium & ~mei.is_valid()
            active_medium &= mei.is_valid()
            
            # handle null and real scatter events (ignore emission events in the medium)
            real_prob = self.index_spectrum(
                mei.sigma_t, channel) / self.index_spectrum(mei.combined_extinction, channel)
            null_prob = 1.0 - real_prob

            null_scatter = (sampler.next_1d(active_medium) < null_prob)
            act_null_scatter |= null_scatter & active_medium
            act_medium_scatter |= ~act_null_scatter & active_medium

            throughput[act_null_scatter] *= mei.sigma_n * dr.rcp(null_prob)
            throughput[act_medium_scatter] *= mei.sigma_s * dr.rcp(real_prob)

            last_scatter_event[act_medium_scatter] = mei
            ray.o[act_null_scatter] = mei.p
            si.t[act_null_scatter] = si.t - mei.t

            # ---------------------- Sampling the Phase Function ----------------------

            phase: mi.PhaseFunction = mei.medium.phase_function()

            # emitter sampling
            sample_emitters = mei.medium.use_emitter_sampling()
            valid_ray |= act_medium_scatter
            specular_chain &= ~act_medium_scatter
            specular_chain |= act_medium_scatter & ~sample_emitters

            active_e = act_medium_scatter & sample_emitters
            emitted, ds = self.sample_emitter(
                mei, scene, sampler, medium, channel, active_e)
            # ds, emitted = scene.sample_emitter_direction(
            #     si, sampler.next_2d(), True, active_e
            # )
            phase_val, phase_pdf = phase.eval_pdf(
                phase_ctx, mei, ds.d, active_e)
            
            result[active_e] += throughput * phase_val * emitted * \
                self.mis_weight(ds.pdf, dr.select(ds.delta, 0, phase_pdf))
            # result[active_e] = throughput * phase_val * emitted

            # phase function sampling
            wo, phase_weight, phase_pdf = phase.sample(phase_ctx, mei, sampler.next_1d(
                act_medium_scatter), sampler.next_2d(act_medium_scatter), act_medium_scatter)
            act_medium_scatter &= (phase_pdf > 0.0)

            # update the throughput
            ray[act_medium_scatter] = mei.spawn_ray(wo)
            throughput[act_medium_scatter] *= phase_weight

            needs_intersection |= act_medium_scatter
            last_scatter_direction_pdf[act_medium_scatter] = phase_pdf
            depth[act_medium_scatter] += 1

            # ---------------------- Surface Interactions ----------------------

            active_surface |= escaped_medium
            # intersect = active_surface & needs_intersection
            # si[intersect] = scene.ray_intersect(ray, intersect)
            si[active_surface] = scene.ray_intersect(ray, active_surface)

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

            emitted, ds = self.sample_emitter(
                si, scene, sampler, medium, channel, active_e)
            # ds, emitted = scene.sample_emitter_direction(
            #     si, sampler.next_2d(), True, active_e
            # )

            wo = si.to_local(ds.d)

            sample1 = sampler.next_1d()
            sample2 = sampler.next_2d()

            bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight = bsdf.eval_pdf_sample(
                bsdf_ctx, si, wo, sample1, sample2
            )

            bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)
            result[active_e] += throughput * bsdf_val * self.mis_weight(
                ds.pdf, dr.select(ds.delta, 0, bsdf_pdf)) * emitted
            # result[active_e] += throughput * bsdf_val * emitted

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

            has_medium_trans = active_surface & si.is_medium_transition()
            medium[has_medium_trans] = si.target_medium(ray.d)

            # ---------------------- Russian Roulette ----------------------

            throughput_max = dr.max(mi.unpolarized_spectrum(throughput))

            rr_prop = dr.minimum(throughput_max, 0.95)
            rr_active = (depth >= self.rr_depth)
            rr_continue = (sampler.next_1d() < rr_prop)

            throughput[rr_active] *= dr.rcp(rr_prop)

            active = (
                active & (active_surface | active_medium) & (
                    ~rr_active | rr_continue) & (dr.neq(throughput_max, 0.0))
            )
            active &= (depth < self.max_depth)

        return result, valid_ray, []

    def sample_emitter(
        self,
        _ref_interaction: mi.Interaction3f,
        scene: mi.Scene,
        sampler: mi.Sampler,
        _medium: mi.MediumPtr,
        channel: mi.UInt32,
        active: bool
    ) -> tuple[mi.Color3f, mi.DirectionSample3f]:

        transmittance = mi.Color3f(1.0)
        active = mi.Bool(active)

        ref_interaction = dr.zeros(mi.Interaction3f)
        ref_interaction[active] = _ref_interaction

        medium = dr.zeros(mi.MediumPtr) 
        medium[active] = _medium

        ds: mi.DirectionSample3f = dr.zeros(mi.DirectionSample3f)
        ds, emitter_val = scene.sample_emitter_direction(
            ref_interaction, sampler.next_2d(active), False, active)
        emitter_val[dr.eq(ds.pdf, 0.0)] = 0.0
        active = active & dr.neq(ds.pdf, 0.0)

        ray = ref_interaction.spawn_ray_to(ds.p)
        max_dist = mi.Float(ray.maxt)

        if isinstance(ref_interaction, mi.SurfaceInteraction3f):
            medium[ref_interaction.is_medium_transition(
            )] = ref_interaction.target_medium(ray.d)

        total_dist = mi.Float(0.0)
        si: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
        needs_intersection = mi.Bool(True)

        loop = mi.Loop("Volpath integrator emitter sampling", lambda: (
            active, ray, total_dist, needs_intersection, medium, si, transmittance))

        loop.set_max_iterations(self.max_depth)

        while loop(active):
            remaining_dist = max_dist - total_dist
            ray.maxt = mi.Float(remaining_dist)
            active &= (remaining_dist > 0.0)

            escaped_medium = mi.Bool(False)
            active_medium = active & dr.neq(medium, None)
            active_surface = active & ~active_medium

            # Sample a medium interaction
            mei: mi.MediumInteraction3f = medium.sample_interaction(
                ray, sampler.next_1d(active_medium), channel, active_medium)
            ray.maxt[active_medium & mei.is_valid()] = dr.minimum(
                mei.t, remaining_dist)
            
            # intersect = needs_intersection & active_medium
            # si[intersect] = scene.ray_intersect(ray, intersect)
            si[active_medium] = scene.ray_intersect(ray, active_medium)

            mei.t[active_medium & (si.t < mei.t)] = dr.inf
            needs_intersection &= ~active_medium

            t = dr.minimum(remaining_dist, dr.minimum(mei.t, si.t)) - mei.mint
            tr = dr.exp(-t * mei.combined_extinction)
            free_flight_pdf = dr.select((si.t < mei.t) | (
                mei.t > remaining_dist), tr, tr * mei.combined_extinction)
            tr_pdf = self.index_spectrum(free_flight_pdf, channel)
            transmittance[active_medium] *= dr.select(
                tr_pdf > 0.0, tr / tr_pdf, 0.0)

            # Handle exceeding the maximum distance by medium sampling
            total_dist[active_medium & (
                mei.t > remaining_dist) & mei.is_valid()] = ds.dist
            mei.t[active_medium & (mei.t > remaining_dist)] = dr.inf

            escaped_medium = active_medium & ~mei.is_valid()
            active_medium &= mei.is_valid()
            total_dist[active_medium] += mei.t

            ray.o[active_medium] = mei.p
            si.t[active_medium] = si.t - mei.t

            transmittance[active_medium] *= mei.sigma_n

            # Handle interactions with surfaces
            # intersect = active_surface & needs_intersection
            # si[intersect] = scene.ray_intersect(ray, intersect)
            active_surface |= escaped_medium
            si[active_surface] = scene.ray_intersect(ray, active_surface)
            # needs_intersection &= ~intersect
            total_dist[active_surface] += si.t

            active_surface &= si.is_valid() & active & ~active_medium
            bsdf: mi.BSDF = si.bsdf(ray)
            bsdf_val = bsdf.eval_null_transmission(si, active_surface)
            bsdf_val = si.to_world_mueller(bsdf_val, si.wi, si.wi)
            transmittance[active_surface] *= bsdf_val

            # Update the ray with new origin & t parameter
            ray[active_surface] = si.spawn_ray(ray.d)
            ray.maxt = mi.Float(remaining_dist)
            needs_intersection |= active_surface

            # Continue tracing through scene if non-zero weights exist
            transmittance_max = dr.max(mi.unpolarized_spectrum(transmittance))
            active &= (active_medium | active_surface) & dr.neq(
                transmittance_max, 0.0)

            # if a medium transition is taking place: Update the medium pointer
            has_medium_trans = active_surface & si.is_medium_transition()
            medium[has_medium_trans] = si.target_medium(ray.d)

        return transmittance * emitter_val, ds


mi.register_integrator("volpath_tracer", lambda props: VolPathTracer(props))
