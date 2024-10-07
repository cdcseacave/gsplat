#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#define NORMALIZE_EPS 1.0E-12F

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Rasterization to Pixels Backward Pass
 ****************************************************************************/

template <uint32_t COLOR_DIM, typename S>
__global__ void rasterize_to_pixels_bwd_radegs_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    // fwd inputs
    const vec2<S> *__restrict__ means2d, // [C, N, 2] or [nnz, 2]
    const vec3<S> *__restrict__ conics,  // [C, N, 3] or [nnz, 3]
    const S *__restrict__ colors,      // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    const S *__restrict__ opacities,   // [C, N] or [nnz]
    const vec2<S> *__restrict__ ray_planes, // [C, N, 2]
    const vec3<S> *__restrict__ normals, // [C, N, 3]
    const S *__restrict__ ts,            // [C, N]
    const S *__restrict__ backgrounds, // [C, COLOR_DIM] or [nnz, COLOR_DIM]
    const bool *__restrict__ masks,    // [C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const S *__restrict__ render_alphas,  // [C, image_height, image_width, 1]
    const S *__restrict__ render_depths,  // [C, image_height, image_width, 1]
    const vec3<S> *__restrict__ render_normals,  // [C, image_height, image_width, 3]
    const int32_t *__restrict__ last_ids, // [C, image_height, image_width]
    const int32_t *__restrict__ max_ids, // [C, image_height, image_width]
    // grad outputs
    const S *__restrict__ v_render_colors, // [C, image_height, image_width,
                                           // COLOR_DIM]
    const S *__restrict__ v_render_alphas, // [C, image_height, image_width, 1]
    const S *__restrict__ v_render_depths, // [C, image_height, image_width, 1]
    const S *__restrict__ v_render_mdepths, // [C, image_height, image_width, 1]
    const S *__restrict__ v_render_normals, // [C, image_height, image_width, 3]
    // grad inputs
    vec2<S> *__restrict__ v_means2d_abs, // [C, N, 2] or [nnz, 2]
    vec2<S> *__restrict__ v_means2d,     // [C, N, 2] or [nnz, 2]
    vec3<S> *__restrict__ v_conics,      // [C, N, 3] or [nnz, 3]
    S *__restrict__ v_colors,   // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    S *__restrict__ v_opacities, // [C, N] or [nnz]
    vec2<S> *__restrict__ v_camera_planes,
    vec2<S> *__restrict__ v_ray_planes,
    vec3<S> *__restrict__ v_normals,
    S *__restrict__ v_ts,
    mat3<S> *__restrict__ K
) {
    auto block = cg::this_thread_block();
    uint32_t camera_id = block.group_index().x;
    uint32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    render_alphas += camera_id * image_height * image_width;
    last_ids += camera_id * image_height * image_width;
    v_render_colors += camera_id * image_height * image_width * COLOR_DIM;
    v_render_alphas += camera_id * image_height * image_width;
    v_render_depths += camera_id * image_height * image_width;
    v_render_normals += camera_id * image_height * image_width * 3;

    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    // when the mask is provided, do nothing and return if
    // this tile is labeled as False
    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    const S px = (S)j + 0.5f;
    const S py = (S)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id =
        min(i * image_width + j, image_width * image_height - 1);

    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec3<S> *xy_opacity_batch =
        reinterpret_cast<vec3<float> *>(&id_batch[block_size]); // [block_size]
    vec3<S> *conic_batch =
        reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]
        );                                         // [block_size]
    S *rgbs_batch = (S *)&conic_batch[block_size]; // [block_size * COLOR_DIM]
    vec2<S> *ray_planes_batch =
        reinterpret_cast<vec2<float> *>(&rgbs_batch[block_size * COLOR_DIM]); // [block_size]
    S *ts_batch = (S *)&ray_planes_batch[block_size]; // [block_size]
    vec3<S> *normals_batch =
        reinterpret_cast<vec3<float> *>(&ts_batch[block_size]); // [block_size]
    vec2<S> *camera_planes_batch =
        reinterpret_cast<vec2<float> *>(&normals_batch[block_size]); // [block_size * 3]


    // this is the T AFTER the last gaussian in this pixel
    S T_final = 1.0f - render_alphas[pix_id];
    S T = T_final;
    // the contribution from gaussians behind the current one
    S buffer[COLOR_DIM] = {0.f};
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    S v_render_c[COLOR_DIM];
    GSPLAT_PRAGMA_UNROLL
    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * COLOR_DIM + k];
    }
    const S v_render_a = v_render_alphas[pix_id];


    const float ddelx_dx = 0.5 * image_width;
    const float ddely_dy = 0.5 * image_height;


    float accum_rec = { 0 };
    float dL_dpixel;
    vec3<S> accum_coord_rec{0, 0, 0};
    vec3<S> dL_dpixel_coord{0, 0, 0};
    float accum_t_rec = 0;
    float accum_alpha_rec = 0;
    vec3<S> accum_normal_rec{0, 0, 0};
    vec3<S> dL_dpixel_mcoord;

    float dL_dt;

    float dL_dpixel_t = 0;
    float dL_dpixel_mt = 0;
    float dL_dalpha = 0;

    const float w_final = inside ? render_alphas[pix_id] : 0;
    const vec2<S> pixf = { (S)j, (S)i }; // TODO: check if order is correct
    const vec2<S> pixnf = {(pixf.x-image_width/2.f)/(*K)[0][0],(pixf.y-image_height/2.f)/(*K)[1][1]};
    const float ln = sqrt(pixnf.x*pixnf.x+pixnf.y*pixnf.y+1);
    vec3<S> dL_dpixel_normal = {0.f, 0.f, 0.f};


    float last_alpha = 0;
    float last_color[COLOR_DIM] = { 0 };
    float last_coord[3] = { 0 };
    float last_t = 0;
    float last_dL_dw = 0;
    vec3<S> last_normal{0, 0, 0};


    if (inside) {
        dL_dalpha = v_render_alphas[pix_id];
        float ww = w_final*w_final;

        {
           float dL_dpixel_depth_w = v_render_depths[pix_id];
           float pixel_accum_depth = render_depths[pix_id] * render_alphas[pix_id];
           dL_dalpha -= dL_dpixel_depth_w*pixel_accum_depth/ww;
           dL_dpixel_t = dL_dpixel_depth_w / w_final / ln;
           dL_dpixel_mt = v_render_mdepths[pix_id] / ln;
        }

        {
            vec3<S> dL_dpixel_normaln;
            dL_dpixel_normaln.x = v_render_normals[pix_id * 3];
            dL_dpixel_normaln.y = v_render_normals[pix_id * 3 + 1];
            dL_dpixel_normaln.z = v_render_normals[pix_id * 3 + 2];

            glm::vec3 normaln = glm::vec3(render_normals[pix_id]);
            S normal_len = glm::length(normaln);
            glm::vec3 dL;
            if (normal_len < NORMALIZE_EPS)
                dL = dL_dpixel_normaln / NORMALIZE_EPS;
            else
                dL = (dL_dpixel_normaln - glm::dot(dL_dpixel_normaln, normaln) * normaln) / normal_len;

            dL_dpixel_normal = dL;
        }
    }

    uint32_t contributor = range_end - range_start;
    const int last_contributor = inside ? last_ids[pix_id] : 0;
    const int max_contributor = inside ? max_ids[pix_id] : 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final =
        cg::reduce(warp, bin_final, cg::greater<int>());
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        // These values can be negative so must be int32 instead of uint32
        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - tr;
        if (idx >= range_start) {
            int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
            id_batch[tr] = g;
            const vec2<S> xy = means2d[g];
            const S opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                rgbs_batch[tr * COLOR_DIM + k] = colors[g * COLOR_DIM + k];
            }
            ray_planes_batch[tr] = ray_planes[g];
            ts_batch[tr] = ts[g];
            normals_batch[tr] = normals[g];
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size;
             ++t) {
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            S alpha;
            S opac;
            vec2<S> delta;
            vec3<S> conic;
            S vis;

            if (valid) {
                conic = conic_batch[t];
                vec3<S> xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = {xy_opac.x - px, xy_opac.y - py};
                S sigma = 0.5f * (conic.x * delta.x * delta.x +
                                  conic.z * delta.y * delta.y) +
                          conic.y * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(0.999f, opac * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = false;
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }
            S v_rgb_local[COLOR_DIM] = {0.f};
            vec3<S> v_conic_local = {0.f, 0.f, 0.f};
            vec2<S> v_xy_local = {0.f, 0.f};
            vec2<S> v_xy_abs_local = {0.f, 0.f};
            S v_opacity_local = 0.f;
            vec2<S> v_ray_plane_local = {0.f, 0.f};
            vec3<S> v_normal_local = {0.f, 0.f, 0.f};
            S v_ts_local = 0.f;
            vec2<S> ray_plane;
            // initialize everything to 0, only set if the lane is valid
            if (valid) {
                // compute the current T for this gaussian
                S ra = 1.0f / (1.0f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const S fac = alpha * T;
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                // contribution from this pixel
                S v_alpha = 0.f;
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    v_alpha +=
                        (rgbs_batch[t * COLOR_DIM + k] * T - buffer[k] * ra) *
                        v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;
                // contribution from background pixel
                if (backgrounds != nullptr) {
                    S accum = 0.f;
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if (opac * vis <= 0.999f) {
                    const S v_sigma = -opac * vis * v_alpha;
                    v_conic_local = {
                        0.5f * v_sigma * delta.x * delta.x,
                        v_sigma * delta.x * delta.y,
                        0.5f * v_sigma * delta.y * delta.y
                    };
                    v_xy_local = {
                        v_sigma * (conic.x * delta.x + conic.y * delta.y),
                        v_sigma * (conic.y * delta.x + conic.z * delta.y)
                    };
                    if (v_means2d_abs != nullptr) {
                        v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                    }
                    v_opacity_local = v_alpha;
                }

                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    buffer[k] += rgbs_batch[t * COLOR_DIM + k] * fac;
                }

                { // Depth
                    const float t_center = ts_batch[t];
                    ray_plane = ray_planes_batch[t];
                    float tt = t_center + (ray_plane.x * delta.x + ray_plane.y * delta.y);
                    accum_t_rec = last_alpha * last_t + (1.f - last_alpha) * accum_t_rec;
                    last_t = tt;
                    v_opacity_local += (t - accum_t_rec) * dL_dpixel_t;
                    dL_dt = fac * dL_dpixel_t;
                    if (contributor == batch_end - max_contributor-1) {
                        dL_dt += dL_dpixel_mt;
                    }
                }

                { // Normal
                    v_ts_local = dL_dt;
                    v_ray_plane_local = {dL_dt * delta.x / (*K)[0][0], dL_dt * delta.y / (*K)[1][1]};

                    vec3<S> normal = normals_batch[t];
                    // Update last color (to be used in the next iteration)
                    accum_normal_rec = last_alpha * last_normal + (1.f - last_alpha) * accum_normal_rec;
                    last_normal = normal;
                    vec3<S> normal_contrib = (normal - accum_normal_rec) * dL_dpixel_normal;
                    v_opacity_local += normal_contrib.x + normal_contrib.y + normal_contrib.z;

                    v_normal_local = fac * dL_dpixel_normal;
                }


            }
            warpSum<COLOR_DIM, S>(v_rgb_local, warp);
            warpSum<decltype(warp), S>(v_conic_local, warp);
            warpSum<decltype(warp), S>(v_xy_local, warp);
            if (v_means2d_abs != nullptr) {
                warpSum<decltype(warp), S>(v_xy_abs_local, warp);
            }
            warpSum<decltype(warp), S>(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t]; // flatten index in [C * N] or [nnz]
                S *v_rgb_ptr = (S *)(v_colors) + COLOR_DIM * g;
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                S *v_conic_ptr = (S *)(v_conics) + 3 * g;
                gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

                S *v_xy_ptr = (S *)(v_means2d) + 2 * g;
                gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if (v_means2d_abs != nullptr) {
                    S *v_xy_abs_ptr = (S *)(v_means2d_abs) + 2 * g;
                    gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                gpuAtomicAdd(v_opacities + g, vis * v_opacity_local);

                gpuAtomicAdd(v_ts + g, v_ts_local);

                S *v_ray_planes_ptr = (S *)(v_ray_planes) + 2 * g;
                gpuAtomicAdd(v_ray_planes_ptr, v_ray_plane_local.x);
                gpuAtomicAdd(v_ray_planes_ptr+1, v_ray_plane_local.y);

                S *v_normals_ptr = (S *)(v_normals) + 3 * g;
                gpuAtomicAdd(v_normals_ptr, v_normal_local.x);
                gpuAtomicAdd(v_normals_ptr + 1, v_normal_local.y);
                gpuAtomicAdd(v_normals_ptr + 2, v_normal_local.z);
            }
        }
    }
}

template <uint32_t CDIM>
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
call_kernel_with_dim(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
    const torch::Tensor &camera_planes,
    const torch::Tensor &ray_planes,
    const torch::Tensor &normals,
    const torch::Tensor &ts,
    const torch::Tensor &K,
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const torch::Tensor &render_depths, // [C, image_height, image_width, 1]
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &render_normals, // [C, image_height, image_width, 3]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    const torch::Tensor &max_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &v_render_depths, // [C, image_height, image_width, 1]
    const torch::Tensor &v_render_mdepths, // [C, image_height, image_width, 1]
    const torch::Tensor &v_render_normals, // [C, image_height, image_width, 3]
    // options
    bool absgrad
) {
    printf("Called rasterize_to_pixels kernel caller\n");

    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(colors);
    GSPLAT_CHECK_INPUT(opacities);
    GSPLAT_CHECK_INPUT(camera_planes);
    GSPLAT_CHECK_INPUT(ray_planes);
    GSPLAT_CHECK_INPUT(normals);
    GSPLAT_CHECK_INPUT(ts);
    GSPLAT_CHECK_INPUT(K);
    GSPLAT_CHECK_INPUT(tile_offsets);
    GSPLAT_CHECK_INPUT(flatten_ids);
    GSPLAT_CHECK_INPUT(render_depths);
    GSPLAT_CHECK_INPUT(render_alphas);
    GSPLAT_CHECK_INPUT(render_normals);
    GSPLAT_CHECK_INPUT(last_ids);
    GSPLAT_CHECK_INPUT(max_ids);
    GSPLAT_CHECK_INPUT(v_render_colors);
    GSPLAT_CHECK_INPUT(v_render_alphas);
    GSPLAT_CHECK_INPUT(v_render_depths);
    GSPLAT_CHECK_INPUT(v_render_mdepths);
    GSPLAT_CHECK_INPUT(v_render_normals);
    if (backgrounds.has_value()) {
        GSPLAT_CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        GSPLAT_CHECK_INPUT(masks.value());
    }

    bool packed = means2d.dim() == 2;

    uint32_t C = tile_offsets.size(0);         // number of cameras
    uint32_t N = packed ? 0 : means2d.size(1); // number of gaussians
    uint32_t n_isects = flatten_ids.size(0);
    uint32_t COLOR_DIM = colors.size(-1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, tile_height, tile_width};

    torch::Tensor v_means2d = torch::zeros_like(means2d);
    torch::Tensor v_conics = torch::zeros_like(conics);
    torch::Tensor v_colors = torch::zeros_like(colors);
    torch::Tensor v_opacities = torch::zeros_like(opacities);
    torch::Tensor v_means2d_abs;
    if (absgrad) {
        v_means2d_abs = torch::zeros_like(means2d);
    }
    torch::Tensor v_camera_planes = torch::zeros_like(camera_planes);
    torch::Tensor v_ray_planes = torch::zeros_like(ray_planes);
    torch::Tensor v_normals = torch::zeros_like(normals);
    torch::Tensor v_ts = torch::zeros_like(ts);

    if (n_isects) {
        const uint32_t shared_mem =
            tile_size * tile_size *
            (sizeof(int32_t) + sizeof(vec3<float>) + sizeof(vec3<float>) +
             sizeof(float) * COLOR_DIM + sizeof(vec2<float>) + sizeof(float) + sizeof(vec3<float>));
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

        if (cudaFuncSetAttribute(
                rasterize_to_pixels_bwd_radegs_kernel<CDIM, float>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shared_mem
            ) != cudaSuccess) {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ",
                shared_mem,
                " bytes), try lowering tile_size."
            );
        }

        rasterize_to_pixels_bwd_radegs_kernel<CDIM, float>
            <<<blocks, threads, shared_mem, stream>>>(
                C,
                N,
                n_isects,
                packed,
                reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                reinterpret_cast<vec2<float> *>(ray_planes.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(normals.data_ptr<float>()),
                ts.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
                image_width,
                image_height,
                tile_size,
                tile_width,
                tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                render_depths.data_ptr<float>(),
                render_alphas.data_ptr<float>(),
                reinterpret_cast<vec3<float> *>(render_normals.data_ptr<float>()),
                last_ids.data_ptr<int32_t>(),
                max_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                v_render_depths.data_ptr<float>(),
                v_render_mdepths.data_ptr<float>(),
                v_render_normals.data_ptr<float>(),
                absgrad ? reinterpret_cast<vec2<float> *>(
                              v_means2d_abs.data_ptr<float>()
                          )
                        : nullptr,
                reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(v_conics.data_ptr<float>()),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                reinterpret_cast<vec2<float> *>(v_camera_planes.data_ptr<float>()),
                reinterpret_cast<vec2<float> *>(v_ray_planes.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(v_normals.data_ptr<float>()),
                v_ts.data_ptr<float>(),
                reinterpret_cast<mat3<float> *>(K.data_ptr<float>())
            );
    }

    return std::make_tuple(
        v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities,
        v_camera_planes, v_ray_planes, v_normals, v_ts
    );
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
rasterize_to_pixels_bwd_radegs_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
    const torch::Tensor &camera_planes,
    const torch::Tensor &ray_planes,
    const torch::Tensor &normals,
    const torch::Tensor &ts,
    const torch::Tensor &K,
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const torch::Tensor &render_depths, // [C, image_height, image_width, 1]
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &render_normals, // [C, image_height, image_width, 3]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    const torch::Tensor &max_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &v_render_depths, // [C, image_height, image_width, 1]
    const torch::Tensor &v_render_mdepths, // [C, image_height, image_width, 1]
    const torch::Tensor &v_render_normals, // [C, image_height, image_width, 3]
    // options
    bool absgrad
) {

    GSPLAT_CHECK_INPUT(colors);
    uint32_t COLOR_DIM = colors.size(-1);

#define __GS__CALL_(N)                                                         \
    case N:                                                                    \
        return call_kernel_with_dim<N>(                                        \
            means2d,                                                           \
            conics,                                                            \
            colors,                                                            \
            opacities,                                                         \
            camera_planes,                                                     \
            ray_planes,                                                        \
            normals,                                                           \
            ts,                                                                \
            K,                                                                 \
            backgrounds,                                                       \
            masks,                                                             \
            image_width,                                                       \
            image_height,                                                      \
            tile_size,                                                         \
            tile_offsets,                                                      \
            flatten_ids,                                                       \
            render_depths,                                                     \
            render_alphas,                                                     \
            render_normals,                                                    \
            last_ids,                                                          \
            max_ids,                                                          \
            v_render_colors,                                                   \
            v_render_alphas,                                                   \
            v_render_depths,                                                   \
            v_render_mdepths,                                                   \
            v_render_normals,                                                  \
            absgrad                                                            \
        );

    switch (COLOR_DIM) {
        __GS__CALL_(1)
        __GS__CALL_(2)
        __GS__CALL_(3)
        __GS__CALL_(4)
        __GS__CALL_(5)
        __GS__CALL_(8)
        __GS__CALL_(9)
        __GS__CALL_(16)
        __GS__CALL_(17)
        __GS__CALL_(32)
        __GS__CALL_(33)
        __GS__CALL_(64)
        __GS__CALL_(65)
        __GS__CALL_(128)
        __GS__CALL_(129)
        __GS__CALL_(256)
        __GS__CALL_(257)
        __GS__CALL_(512)
        __GS__CALL_(513)
    default:
        AT_ERROR("Unsupported number of channels: ", COLOR_DIM);
    }
}

} // namespace gsplat
