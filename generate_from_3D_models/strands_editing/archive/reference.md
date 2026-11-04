# Feature Specification: Interactive Hair-Strand Brushing / Local Deformation

## 1. Objective

Implement an interactive hair-brushing feature for a 3D hairstyling application. When the user drags a point or brushes a region of a hair strand, nearby parts of the strand should move smoothly and naturally.

Priorities:
1. Visually plausible deformation rather than physically accurate simulation.
2. Fast interactive response.
3. Fixed hair roots.
4. Preserve existing strand shape as much as possible.
5. Preserve highlight/dye patterns attached to each strand.
6. Keep the implementation simple and extensible.

## 2. Required First Step: Inspect the Existing Codebase

Before implementing anything, inspect the current geometry representation, curve evaluation, interaction handling, and rendering pipeline.

Determine:
- Whether strands are represented by Bézier curves, B-splines, Catmull–Rom splines, polylines, or another representation.
- Whether stored points are interpolation points, spline control points, or sampled curve positions.
- How hair roots are identified and anchored.
- Whether strands are independent curves or generated from guide curves.
- How strand colors and highlights are represented.
- Whether deformation updates can reuse existing GPU buffers or curve data structures.

Reuse existing architecture wherever possible. Do not introduce a new physics framework or replace the current curve representation without a clear technical reason.

## 3. Recommended Baseline Algorithm

Use a local Gaussian displacement field as the first implementation, adapted to the existing curve representation.

Let the original strand be represented by points or curve parameters indexed by i. Let k denote the selected location and Δ the requested displacement.

Define d(i,k) as the distance along the original strand between the selected location and the location associated with i.

Use the Gaussian influence weight:

    w_i = exp(-d(i,k)^2 / (2*sigma^2))

Then calculate:

    P_i_new = P_i_original + w_i * Delta

Requirements:
- The selected location should follow the drag target.
- The root must remain exactly fixed.
- Influence should decay smoothly with distance along the strand.
- Distant portions should move negligibly.
- Use the original geometry captured at drag start as the deformation reference.
- Recompute the deformation from that reference on every drag update, rather than accumulating displacement from the previous frame.
- Handle open curves and their endpoints correctly.

For a fixed root, explicitly exclude the root from displacement and restore its exact original position after each update.

### Curve-representation considerations

Do not blindly apply the formula to arbitrary spline control points.

If the current points lie on the curve, arc-length-based point weights can be used directly.

If the curve uses Bézier or B-spline control points, derive the influence of each control point from the curve's native parameterization, for example using its associated parameter location, or use an appropriate local curve-deformation method. Ensure the resulting evaluated curve follows the intended drag motion.

If necessary, sample the curve by approximate arc length to construct a distance lookup table. Avoid repeatedly evaluating expensive arc-length integrals during mouse movement.

## 4. Interaction Model

At the beginning of a drag:
1. Identify the selected strand and closest curve parameter.
2. Store the original curve/control-point state.
3. Determine the fixed root.
4. Cache the influence weights or other reusable data if appropriate.

During dragging:
1. Calculate the desired displacement from the original selected location to the current target.
2. Apply the local displacement field to the original geometry.
3. Enforce the root constraint.
4. Update the curve representation and render the result.

At drag release:
- Commit the final geometry as the new strand state.
- Preserve the new shape until another edit is performed.
- Ensure undo/redo can be integrated with the application's existing editing architecture, if available.

Do not automatically restore the original geometry after the mouse is released.

## 5. Preserve Highlights and Strand Attributes

Highlights, dye patterns, and other strand-specific attributes should remain attached to the strand rather than fixed in world space.

If color is defined as a function of strand parameter u, preserve that function while modifying the geometry. For example, a highlight at u = 0.6 should remain at the corresponding location along the deformed strand.

Avoid regenerating or interpolating colors unnecessarily if the existing representation already supports stable per-strand parameters.

## 6. Performance Requirements

The implementation should prioritize low per-frame CPU overhead.

- Precompute arc-length information and influence weights when possible.
- Cache reusable geometry and curve-evaluation data.
- Update only strands affected by the interaction.
- Avoid full nonlinear optimization, matrix inversion, or physical simulation in the baseline implementation.
- Avoid unnecessary allocations during drag updates.
- Reuse existing rendering buffers where practical.
- Preserve the existing rendering quality and application architecture.

The baseline deformation should be O(n) per affected strand, where n is the number of points or control points updated. Spatial selection across many strands should use an existing spatial index if one is available, or introduce a simple spatial query only if needed.

Do not promise a specific speedup without benchmarking the actual application.

## 7. Optional Improvements — Only If Needed

After the baseline is working, visually evaluate the result and profile performance.

If the deformation looks unnatural, consider adding one or more of the following:

1. **Compact-support influence kernel:** Make the displacement exactly zero beyond a chosen arc-length radius.
2. **Length preservation:** Apply a small, fixed number of local constraint-projection iterations to reduce unintended stretching.
3. **Curvature preservation:** Add a lightweight correction that preserves the strand's original bending characteristics.
4. **Guide-curve deformation:** If many strands are generated from guide curves, deform guides and regenerate or update dependent strands instead of solving each strand independently.
5. **Dynamic response:** Add inertia and damping only if the desired interaction requires visible swinging or bounce.

Do not implement all these features preemptively. Keep the first version simple.

## 8. Validation and Acceptance Criteria

Test the implementation with representative strands, including straight, curved, tightly bent, and unevenly sampled curves.

Verify:
- The root position never changes.
- Dragging the selected location produces the expected displacement.
- Nearby locations move smoothly, with influence decreasing along the strand.
- The curve remains visually smooth.
- Repeated drag updates do not accumulate unintended deformation.
- Existing curl and shape are reasonably preserved.
- Highlight patterns remain attached to their strands.
- Releasing the mouse retains the edited geometry.
- Performance remains acceptable with the application's representative strand count.

Record baseline and post-implementation timings for the deformation update and, where practical, curve evaluation and rendering separately.

## 9. Deliverables

1. Implement the simplest compatible baseline in the existing codebase.
2. Explain which files and functions were changed.
3. Describe how the algorithm fits the existing curve representation.
4. Report any limitations, especially around spline control points, root constraints, and shape preservation.
5. Provide a small benchmark or profiling result using representative application data.
6. Recommend further optimization only if profiling or visual inspection indicates a need.

The primary goal is a fast, visually acceptable interactive hair-brushing tool, not a physically accurate hair simulator.