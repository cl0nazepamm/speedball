# Emissive surface sampling for probe GI

Status: proposed follow-up; not implemented. Reference: Situation Monitoring room, emissive-only lighting.

## Objective

Thin wall strips, ceiling panels, and monitors should contribute predictably at different probe densities. Increasing quality should primarily reduce noise and improve detail, rather than accidentally removing a strip's illumination. Keep the scene emissive-only and preserve shared geometry/materials and instancing.

## Confirmed starting points

- The reference room has no active traced lights or injected sky. Its emissives currently reach probes through discovery rays; the explicit emitter list was empty during diagnosis.
- At the reference bounds and default padding, Divisions 14/15/16 create 975/1,120/1,530 probes. The first interior Y layer moves from 30.8/36.0/2.4 cm. Placement, visibility, cell-scaled bias, and update cadence therefore confound a density comparison.
- `spectral_scene.js::collectEmitterRecords` is opt-in through `giEmitter`, approximates emitters with bounding spheres, and currently reads only the first material. Its geometry/area collection does not expand instance transforms.
- `gi_probes.js::emitterVisKernel` and `injectEmitterVirtualRays` skip probes inside an emitter's bounding sphere. This fails for long strips surrounding nearby probes. The selected emitter cap is 16.
- Tagged emitters have traced emission zeroed to avoid double counting. Replacing that policy requires coordinated changes to discovery hits, explicit sampling, diffuse history, and reflection history.

## Implementation order

1. **Capture a reproducible baseline.** Save scene/export, camera, exposure, emission values, and GI settings. Compare 14/15/16 at identical padding and both equal elapsed time and equal samples per probe. Add isolated strip, panel, textured screen, and occluder scenes. Use a converged reference to distinguish lost energy from coarse-grid light leaking; the brighter result is not automatically the reference.

2. **Build an instance-aware emitter distribution.** Reuse BLAS triangle data and material IDs. Store emitting triangle references plus instance transforms, rather than duplicating geometry. Respect active instance count, material groups, draw ranges, visibility, sidedness, emission factors, emissive textures, and alpha. Weight selection by emitted power and world-space area. Track transform, material, texture, and topology invalidations through existing scene update APIs.

3. **Sample actual emitting surfaces.** Begin with a small fixed number of power-weighted emitter samples per updated probe, using a CDF/alias distribution that can select any emitter instead of silently dropping everything after 16. Sample triangle points with a known area PDF and convert to solid angle. Trace visibility to the sampled point with scale-aware endpoint handling. Nearby probes remain valid; remove the enclosing-sphere exclusion. Long strips need no hand-authored light proxies.

4. **Define the estimator before integrating it.** Derive directional radiance and cosine-convolved irradiance normalization for both uniform discovery rays and emitter samples. Include emitter selection probability, triangle probability, area-to-solid-angle Jacobian, and sidedness. Combine estimators with MIS and account for emissive discovery hits exactly once. Preserve misses/environment behavior and unselected-emitter support. Do not add virtual samples to the existing denominator without deriving the resulting energy normalization.

5. **Integrate diffuse and reflection caches.** Feed weighted contributions into the existing directional caches and temporal moments, keeping spatial filtering separate from accumulation. Verify rough and glossy lobes independently: emission must not disappear when a surface is explicitly sampled, nor become brighter from double counting. Keep High's complete-tile updates. Respect WebGPU storage-binding limits and the existing shared scene resource lifecycle.

6. **Bound the cost.** Budget emitter visibility rays explicitly alongside discovery rays. Quality presets choose sampling effort; avoid a new main-panel slider. First deliver correct bounded sampling, then consider spatial selection or reservoir reuse only if measurements justify it. Track dispatch time, memory, selected-emitter coverage, and convergence.

7. **Stabilize placement separately.** Decouple auto-padding/bounds from the quality budget, validate probe clearance near surfaces, and improve relocation/classification. Retain explicit custom-volume bounds and resolution. Evaluate this independently so placement improvements do not conceal estimator errors.

8. **Integrate max.js and snapshots.** Ship the same Speedball code and settings in the live viewer and export. Preserve legacy projects and manual emitter overrides; define an explicit migration policy for old `giEmitter` suppression. Automatic emissive selection should not require authors to add conventional lights or tag every strip. Keep artistic GI strength and exposure separate from sampling quality.

## Acceptance checks

- Long thin strip illuminates nearby visible probes, including probes inside its former bounding sphere. An intervening wall blocks it.
- More than 16 emitters, repeated instances, nonuniform transforms, multiple materials, and emissive textures retain their contributions.
- Emission scaling is linear; combined sampling agrees with a converged reference within measured Monte Carlo error. No duplicate energy between discovered and explicitly sampled emission.
- At fixed field bounds, moving between nearby densities preserves the strip's contribution after convergence. Any remaining reconstruction/occlusion error is measured separately.
- Test gated and Monte Carlo sampling, one/two cascades, all reflection tiers, live edits, and custom volumes. No new seams, NaNs, flicker, or stale emitter transforms.
- Live viewer and re-exported runtime agree. Settings survive save/reload and legacy defaults remain intentional.
- Record frame cost and convergence against the current implementation on the reference GPU before enabling automatic emitter sampling by default.

Reference: PBRT, “A Better Path Tracer” and “Light Sampling” (light sampling PDFs and MIS): https://www.pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Better_Path_Tracer and https://www.pbr-book.org/4ed/Light_Sources/Light_Sampling.
