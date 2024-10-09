#include "bindings.h"
#include "helpers.cuh"
#include "utils.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/pca.hpp>
#include "auxiliary.h"


namespace gsplat {

    namespace cg = cooperative_groups;

template <typename T>
inline __device__ void persp_proj_vjp_rade(
        // fwd inputs
        const vec3<T> mean3d,
        const mat3<T> cov3d,
        const T fx,
        const T fy,
        const T cx,
        const T cy,
        const uint32_t width,
        const uint32_t height,
        const T *view_matrix,
        const vec4<T> conic_opacity,
        const vec2<T> *v_camera_planes,
        const vec2<T> v_ray_plane,
        const vec3<T> v_normal,
        const vec4<T> v_conic,
        // grad outputs
        const mat2<T> v_cov2d,
        const vec2<T> v_mean2d,
        // grad inputs
        vec3<T> &v_mean3d,
        mat3<T> &v_cov3d,
        T &v_opacity
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    T tan_fovx = 0.5f * width / fx;
    T tan_fovy = 0.5f * height / fy;
    T lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    T lim_x_neg = cx / fx + 0.3f * tan_fovx;
    T lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    T lim_y_neg = cy / fy + 0.3f * tan_fovy;

    T rz = 1.f / z;
    T rz2 = rz * rz;
    T tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    T ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
            fx * rz,
            0.f, // 1st column
            0.f,
            fy * rz, // 2nd column
            -fx * tx * rz2,
            -fy * ty * rz2 // 3rd column
    );

    mat3<T> Jzero = mat3<T>(
            fx * rz,
            0.f,
            0.f, // 1st column
            0.f,
            fy * rz,
            0.f, // 2nd column
            -fx * tx * rz2,
            -fy * ty * rz2,
            0.f  // 3rd column
    );

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    // df/dx = fx * rz * df/dpixx
    // df/dy = fy * rz * df/dpixy
    // df/dz = - fx * mean.x * rz2 * df/dpixx - fy * mean.y * rz2 * df/dpixy
    v_mean3d += vec3<T>(
            fx * rz * v_mean2d[0],
            fy * rz * v_mean2d[1],
            -(fx * x * v_mean2d[0] + fy * y * v_mean2d[1]) * rz2
    );

    // df/dx = -fx * rz2 * df/dJ_02
    // df/dy = -fy * rz2 * df/dJ_12
    // df/dz = -fx * rz2 * df/dJ_00 - fy * rz2 * df/dJ_11
    //         + 2 * fx * tx * rz3 * df/dJ_02 + 2 * fy * ty * rz3
    T rz3 = rz2 * rz;
    mat3x2<T> v_J = v_cov2d * J * glm::transpose(cov3d) +
                    glm::transpose(v_cov2d) * J * cov3d;

    // fov clipping
    if (x * rz <= lim_x_pos && x * rz >= -lim_x_neg) {
        v_mean3d.x += -fx * rz2 * v_J[2][0];
    } else {
        v_mean3d.z += -fx * rz3 * v_J[2][0] * tx;
    }
    if (y * rz <= lim_y_pos && y * rz >= -lim_y_neg) {
        v_mean3d.y += -fy * rz2 * v_J[2][1];
    } else {
        v_mean3d.z += -fy * rz3 * v_J[2][1] * ty;
    }
    v_mean3d.z += -fx * rz2 * v_J[0][0] - fy * rz2 * v_J[1][1] +
                  2.f * fx * tx * rz3 * v_J[2][0] +
                  2.f * fy * ty * rz3 * v_J[2][1];


    vec3<T> t = {mean3d[0], mean3d[1], mean3d[2]};
    T h_x = fx;
    T h_y = fy;
    mat3<T> cov3D = cov3d;
    vec2<T> dL_camera_plane0 = v_camera_planes[0];
    vec2<T> dL_camera_plane1 = v_camera_planes[1];
    vec2<T> dL_camera_plane2 = v_camera_planes[2];

    auto& dL_dray_plane = v_ray_plane;
    auto& dL_dnormal = v_normal;
    auto& dL_dconic = v_conic;
    auto& dL_dopacity = v_opacity;
    auto& dL_dcov = v_cov3d;

    int kernel_size = 1;

    const T limx = 1.3f * tan_fovx;
    const T limy = 1.3f * tan_fovy;
    T txtz = t.x / t.z;
    T tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    const T x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
    const T y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

    txtz = t.x / t.z;
    tytz = t.y / t.z;

    mat3<T> W = mat3<T>(
            view_matrix[0], view_matrix[4], view_matrix[8],
            view_matrix[1], view_matrix[5], view_matrix[9],
            view_matrix[2], view_matrix[6], view_matrix[10]);

    mat3<T> Vrk = cov3d;

    mat3<T> TR = W * Jzero;

    mat3<T> cov2D = glm::transpose(TR) * glm::transpose(Vrk) * TR;

    const T det_0 = max(1e-6, cov2D[0][0] * cov2D[1][1] - cov2D[0][1] * cov2D[0][1]);
    const T det_1 = max(1e-6, (cov2D[0][0] + kernel_size) * (cov2D[1][1] + kernel_size) - cov2D[0][1] * cov2D[0][1]);
    // sqrt here
    const T coef = sqrt(det_0 / (det_1+1e-6) + 1e-6);
    // const T coef = 1.0f;

    mat3<T> Vrk_eigen_vector;
    vec3<T> Vrk_eigen_value;
    float D = glm::findEigenvaluesSymReal(Vrk,Vrk_eigen_value,Vrk_eigen_vector);

    unsigned int min_id = Vrk_eigen_value[0]>Vrk_eigen_value[1]? (Vrk_eigen_value[1]>Vrk_eigen_value[2]?2:1):(Vrk_eigen_value[0]>Vrk_eigen_value[2]?2:0);

    mat3<T> Vrk_inv;
    vec3<T> eigenvector_min;
    bool well_conditioned = Vrk_eigen_value[min_id]>0.00000001;
    if(well_conditioned)
    {
        mat3<T> diag = mat3<T>(1/Vrk_eigen_value[0],0,0,
                                   0,1/Vrk_eigen_value[1],0,
                                   0,0,1/Vrk_eigen_value[2]);
        Vrk_inv = Vrk_eigen_vector * diag * glm::transpose(Vrk_eigen_vector);
    }
    else
    {
        eigenvector_min = Vrk_eigen_vector[min_id];
        Vrk_inv = glm::outerProduct(eigenvector_min,eigenvector_min);
    }

    // mat3<T> Vrk_inv = Vrk_eigen_vector * diag * glm::transpose(Vrk_eigen_vector);
    mat3<T> cov_cam_inv = glm::transpose(W) * Vrk_inv * W;
    vec3<T> uvh = {txtz, tytz, 1};
    vec3<T> uvh_m = cov_cam_inv * uvh;
    vec3<T> uvh_mn = glm::normalize(uvh_m);


    T u2 = txtz * txtz;
    T v2 = tytz * tytz;
    T uv = txtz * tytz;

    mat3<T> dL_dVrk;
    vec3<T> plane;
    T dL_du;
    T dL_dv;
    T dL_dl;
    T l;
    T nl;
    vec3<T> dL_duvh;
    mat3<T> dL_dnJ_inv;
    T dL_dnl;
    mat3<T> nJ;
    vec3<T> ray_normal_vector;
    vec3<T> cam_normal_vector;
    vec3<T> normal_vector;
    vec3<T> dL_dnormal_lv;

    // go inside else
        T vb = glm::dot(uvh_m, uvh);
        T vbn = glm::dot(uvh_mn, uvh);

        l = sqrt(t.x*t.x+t.y*t.y+t.z*t.z);
        nJ = mat3<T>(
                1 / t.z, 0.0f, -(t.x) / (t.z * t.z),
                0.0f, 1 / t.z, -(t.y) / (t.z * t.z),
                t.x/l, t.y/l, t.z/l);

        mat3<T> nJ_inv = mat3<T>(
                v2 + 1,	-uv, 		0,
                -uv,	u2 + 1,		0,
                -txtz,	-tytz,		0
        );

        T clamp_vb = max(vb, 0.0000001f);
        T clamp_vbn = max(vbn, 0.0000001f);
        nl = u2+v2+1;
        T factor_normal = l / nl;
        vec3<T> uvh_m_vb = uvh_mn/clamp_vbn;
        plane = nJ_inv * uvh_m_vb;

        glm::vec2 camera_plane0 = {(-(v2 + 1)*t.z+plane[0]*t.x)/nl, (uv*t.z+plane[1]*t.x)/nl};
        glm::vec2 camera_plane1 = {(uv*t.z+plane[0]*t.y)/nl, (-(u2 + 1)*t.z+plane[1]*t.y)/nl};
        glm::vec2 camera_plane2 = {(t.x+plane[0]*t.z)/nl, (t.y+plane[1]*t.z)/nl};

        glm::vec2 ray_plane = {plane[0]*factor_normal, plane[1]*factor_normal};

        ray_normal_vector = {-plane[0]*factor_normal, -plane[1]*factor_normal, -1};

        cam_normal_vector = nJ * ray_normal_vector;
        normal_vector = glm::normalize(cam_normal_vector);
        T lv = glm::length(cam_normal_vector);
        dL_dnormal_lv = dL_dnormal/lv;
        vec3<T> dL_dcam_normal_vector = dL_dnormal_lv - normal_vector * glm::dot(normal_vector,dL_dnormal_lv);
        vec3<T> dL_dray_normal_vector = glm::transpose(nJ) * dL_dcam_normal_vector;

    mat3<T> dL_dnJ;
    if(isnan(uvh_mn.x)||D==0)
    {
        dL_dVrk = mat3<T>(0,0,0,0,0,0,0,0,0);
        dL_dnJ = mat3<T>(0,0,0,0,0,0,0,0,0);
        plane = vec3<T>(0,0,0);
        nl = 1;
        l = 1;
        dL_du = 0;
        dL_dv = 0;
        dL_dl = 0;
    }
    else
    {
        dL_dnJ = glm::outerProduct(dL_dcam_normal_vector,ray_normal_vector);
        dL_dl = (-plane[0] * dL_dray_normal_vector.x - plane[1] * dL_dray_normal_vector.y
                 + plane[0] * dL_dray_plane.x + plane[1] * dL_dray_plane.y) / nl;

        glm::vec2 dL_dplane = glm::vec2(
                (t.x*dL_camera_plane0.x + t.y*dL_camera_plane1.x + t.z*dL_camera_plane2.x
                 -l * dL_dray_normal_vector[0] + dL_dray_plane.x * l) / nl,
                (t.x*dL_camera_plane0.y + t.y*dL_camera_plane1.y + t.z*dL_camera_plane2.y
                 -l * dL_dray_normal_vector[1] + dL_dray_plane.y * l) / nl
        );
        vec3<T> dL_dplane_append = vec3<T>(dL_dplane.x, dL_dplane.y, 0);

        dL_dnl = (0
                -dL_camera_plane0.x * camera_plane0.x - dL_camera_plane0.y * camera_plane0.y
                -dL_camera_plane1.x * camera_plane1.x - dL_camera_plane1.y * camera_plane1.y
                -dL_camera_plane2.x * camera_plane2.x - dL_camera_plane2.y * camera_plane2.y
                - dL_dray_normal_vector[0] * ray_normal_vector.x
                - dL_dray_normal_vector[1] * ray_normal_vector.y
                -dL_dray_plane.x * ray_plane.x - dL_dray_plane.y * ray_plane.y
                ) / nl;


        T tmp = dL_dplane.x * plane.x + dL_dplane.y * plane.y;

        vec3<T> W_uvh = W * uvh;

        if(well_conditioned){
            dL_dVrk = - glm::outerProduct(Vrk_inv * W_uvh, (Vrk_inv/clamp_vb) * (W_uvh * (-tmp) + W * glm::transpose(nJ_inv) * dL_dplane_append));
        }
        else{
            dL_dVrk = mat3<T>(0,0,0,0,0,0,0,0,0);
            T dL_dvb = -tmp / clamp_vb;
            vec3<T> nJ_inv_dL_dplane = glm::transpose(nJ_inv) * vec3<T>(dL_dplane.x / clamp_vb, dL_dplane.y / clamp_vb, 0);
            mat3<T> dL_dVrk_inv = glm::outerProduct(W_uvh, W_uvh * dL_dvb + W * nJ_inv_dL_dplane);
            vec3<T> dL_dv = (dL_dVrk_inv + glm::transpose(dL_dVrk_inv)) * eigenvector_min;
            for(int j =0;j<3;j++)
            {
                if(j!=min_id)
                {
                    T scale = glm::dot(Vrk_eigen_vector[j],dL_dv)/min(Vrk_eigen_value[min_id] - Vrk_eigen_value[j], - 0.0000001f);
                    dL_dVrk += glm::outerProduct(Vrk_eigen_vector[j] * scale, eigenvector_min);
                }
            }
        }


        dL_duvh = 2 * (-tmp) * uvh_m_vb + (cov_cam_inv/clamp_vb) * glm::transpose(nJ_inv) * dL_dplane_append;

        dL_dnJ_inv = glm::outerProduct(dL_dplane_append, uvh_m_vb);

        dL_du = 0
                + dL_dnl
                * 2 * txtz
                + dL_duvh.x
                + (dL_dnJ_inv[0][1] + dL_dnJ_inv[1][0]) * (-tytz) + 2 * dL_dnJ_inv[1][1] * txtz - dL_dnJ_inv[2][0]
                + (dL_camera_plane0.y * t.y + dL_camera_plane1.x * t.y + dL_camera_plane1.y * (-2*t.x)) / nl
                ;

        dL_dv = dL_dnl * 2 * tytz
                + dL_duvh.y
                + (dL_dnJ_inv[0][1] + dL_dnJ_inv[1][0]) * (-txtz) + 2 * dL_dnJ_inv[0][0] * tytz - dL_dnJ_inv[2][1]
                + (dL_camera_plane0.x * (-2*t.y) + dL_camera_plane0.y * t.x + dL_camera_plane1.x * t.x) / nl;;
    }

    const T combined_opacity = conic_opacity[3];
    const T opacity = combined_opacity / (coef + 1e-6);
    const T dL_dcoef = dL_dopacity * opacity;
    const T dL_dsqrtcoef = dL_dcoef * 0.5 * 1. / (coef + 1e-6);
    const T dL_ddet0 = dL_dsqrtcoef / (det_1+1e-6);
    const T dL_ddet1 = dL_dsqrtcoef * det_0 * (-1.f / (det_1 * det_1 + 1e-6));
    const T dcoef_da = dL_ddet0 * cov2D[1][1] + dL_ddet1 * (cov2D[1][1] + kernel_size);
    const T dcoef_db = dL_ddet0 * (-2. * cov2D[0][1]) + dL_ddet1 * (-2. * cov2D[0][1]);
    const T dcoef_dc = dL_ddet0 * cov2D[0][0] + dL_ddet1 * (cov2D[0][0] + kernel_size);
    //TODO gradient is zero if det_0 or det_1 < 0
    // Use helper variables for 2D covariance entries. More compact.
    T a = cov2D[0][0] + kernel_size;
    T b = cov2D[0][1];
    T c = cov2D[1][1] + kernel_size;

    T denom = a * c - b * b;
    T dL_da = 0, dL_db = 0, dL_dc = 0;

    float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

    if (denom2inv != 0)
    {
        // Gradients of loss w.r.t. entries of 2D covariance matrix,
        // given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
        // e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
        dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
        dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
        dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

        if (det_0 <= 1e-6 || det_1 <= 1e-6){
            dL_dopacity = 0;
        } else {
            // Gradiends of alpha respect to conv due to low pass filter
            dL_da += dcoef_da;
            dL_dc += dcoef_dc;
            dL_db += dcoef_db;

            // update dL_dopacity
            dL_dopacity = dL_dopacity * coef;
        }

        // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
        // given gradients w.r.t. 2D covariance matrix (diagonal).
        // cov2D = transpose(T) * transpose(Vrk) * T;
//		dL_dcov[0][0] = (TR[0][0] * TR[0][0] * dL_da + TR[0][0] * TR[1][0] * dL_db + TR[1][0] * TR[1][0] * dL_dc);
//		dL_dcov[1][1] = (TR[0][1] * TR[0][1] * dL_da + TR[0][1] * TR[1][1] * dL_db + TR[1][1] * TR[1][1] * dL_dc);
//		dL_dcov[2][2] = (TR[0][2] * TR[0][2] * dL_da + TR[0][2] * TR[1][2] * dL_db + TR[1][2] * TR[1][2] * dL_dc);
//
//                // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
//                // given gradients w.r.t. 2D covariance matrix (off-diagonal).
//                // Off-diagonal elements appear twice --> double the gradient.
//                // cov2D = transpose(TR) * transpose(Vrk) * TR;
//                dL_dcov[0][1] = 2 * TR[0][0] * TR[0][1] * dL_da + (TR[0][0] * TR[1][1] + TR[0][1] * TR[1][0]) * dL_db + 2 * TR[1][0] * TR[1][1] * dL_dc;
//                dL_dcov[0][2] = 2 * TR[0][0] * TR[0][2] * dL_da + (TR[0][0] * TR[1][2] + TR[0][2] * TR[1][0]) * dL_db + 2 * TR[1][0] * TR[1][2] * dL_dc;
//                dL_dcov[1][2] = 2 * TR[0][2] * TR[0][1] * dL_da + (TR[0][1] * TR[1][2] + TR[0][2] * TR[1][1]) * dL_db + 2 * TR[1][1] * TR[1][2] * dL_dc;
//
//                dL_dcov[1][0] = dL_dcov[0][1];
//                dL_dcov[2][0] = dL_dcov[0][2];
//                dL_dcov[2][1] = dL_dcov[1][2];
    }
    else
    {
        for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
            dL_dcov[i][j] = 0;
    }
////        for (int i = 0; i < 3; i++)
////            for (int j = 0; j < 3; j++)
////                dL_dcov[i][j] += dL_dVrk[i][j];

//	dL_dcov[0][0] += dL_dVrk[0][0];
//	dL_dcov[1][1] += dL_dVrk[1][1];
//	dL_dcov[2][2] += dL_dVrk[2][2];
//	dL_dcov[1] += dL_dVrk[0][1] + dL_dVrk[1][0];
//	dL_dcov[2] += dL_dVrk[0][2] + dL_dVrk[2][0];
//	dL_dcov[4] += dL_dVrk[1][2] + dL_dVrk[2][1];

    dL_dcov += dL_dVrk;

    // Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix TR
    // cov2D = transpose(TR) * transpose(Vrk) * TR;
    T dL_dT00 = 2 * (TR[0][0] * Vrk[0][0] + TR[0][1] * Vrk[0][1] + TR[0][2] * Vrk[0][2]) * dL_da +
                    (TR[1][0] * Vrk[0][0] + TR[1][1] * Vrk[0][1] + TR[1][2] * Vrk[0][2]) * dL_db;
    T dL_dT01 = 2 * (TR[0][0] * Vrk[1][0] + TR[0][1] * Vrk[1][1] + TR[0][2] * Vrk[1][2]) * dL_da +
                    (TR[1][0] * Vrk[1][0] + TR[1][1] * Vrk[1][1] + TR[1][2] * Vrk[1][2]) * dL_db;
    T dL_dT02 = 2 * (TR[0][0] * Vrk[2][0] + TR[0][1] * Vrk[2][1] + TR[0][2] * Vrk[2][2]) * dL_da +
                    (TR[1][0] * Vrk[2][0] + TR[1][1] * Vrk[2][1] + TR[1][2] * Vrk[2][2]) * dL_db;
    T dL_dT10 = 2 * (TR[1][0] * Vrk[0][0] + TR[1][1] * Vrk[0][1] + TR[1][2] * Vrk[0][2]) * dL_dc +
                    (TR[0][0] * Vrk[0][0] + TR[0][1] * Vrk[0][1] + TR[0][2] * Vrk[0][2]) * dL_db;
    T dL_dT11 = 2 * (TR[1][0] * Vrk[1][0] + TR[1][1] * Vrk[1][1] + TR[1][2] * Vrk[1][2]) * dL_dc +
                    (TR[0][0] * Vrk[1][0] + TR[0][1] * Vrk[1][1] + TR[0][2] * Vrk[1][2]) * dL_db;
    T dL_dT12 = 2 * (TR[1][0] * Vrk[2][0] + TR[1][1] * Vrk[2][1] + TR[1][2] * Vrk[2][2]) * dL_dc +
                    (TR[0][0] * Vrk[2][0] + TR[0][1] * Vrk[2][1] + TR[0][2] * Vrk[2][2]) * dL_db;

    // Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
    // TR = W * J
    T dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
    T dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
    T dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
    T dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

    T tz = 1.f / t.z;
    T tz2 = tz * tz;
    T tz3 = tz2 * tz;

    // mat3<T> nJ = mat3<T>(
    // 		1 / t.z, 0.0f, -(t.x) / (t.z * t.z),
    // 		0.0f, 1 / t.z, -(t.y) / (t.z * t.z),
    // 		t.x/l, t.y/l, t.z/l);
    T l3 = l * l * l;

    T dL_dtx =
            x_grad_mul * (
//                        -h_x * tz2 * dL_dJ02 +
            0
                    + dL_du  * tz
                       -dL_dnJ[0][2]*tz2 + dL_dnJ[2][0]*(1/l-t.x*t.x/l3) + dL_dnJ[2][1]*(-t.x*t.y/l3) + dL_dnJ[2][2]*(-t.x*t.z/l3) //this line is from normal
                        +(dL_camera_plane0.x * plane[0] + dL_camera_plane0.y * plane[1] + dL_camera_plane2.x)/nl
                        +dL_dl*t.x/l
                         );
    T dL_dty =
            y_grad_mul * (
//                        -h_y * tz2 * dL_dJ12 +
                    0
                        + dL_dv * tz
                        -dL_dnJ[1][2]*tz2 + dL_dnJ[2][0]*(-t.x*t.y/l3) + dL_dnJ[2][1]*(1/l-t.y*t.y/l3) + dL_dnJ[2][2]*(-t.y*t.z/l3) //this line is from normal
                         +(dL_camera_plane1.x * plane[0] + dL_camera_plane1.y * plane[1] + dL_camera_plane2.y)/nl
                         +dL_dl*t.y/l
                         );
    T dL_dtz = 0
//                -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12
                   - (dL_du * t.x + dL_dv * t.y) * tz2
                   + (dL_dnJ[0][0] + dL_dnJ[1][1]) * (-tz2) + dL_dnJ[0][2] * (2*t.x*tz3) + dL_dnJ[1][2] * (2*t.y*tz3) // two lines are from normal
                   + (dL_dnJ[2][0]*t.x+dL_dnJ[2][1]*t.y)*(-t.z/l3) + dL_dnJ[2][2]*(1/l-t.z*t.z/l3) // two lines are from normal
                   + (dL_camera_plane0.x * (-(v2 + 1)) + dL_camera_plane0.y * uv + dL_camera_plane1.x * uv + dL_camera_plane1.y * (-(u2+1)) + dL_camera_plane2.x * plane[0] + dL_camera_plane2.y * plane[1])/nl
                   + dL_dl*t.z/l
                 ;

    v_mean3d += vec3<T>(dL_dtx, dL_dty, dL_dtz);
}
/****************************************************************************
 * Projection of Gaussians (Single Batch) Backward Pass
 ****************************************************************************/

template <typename T>
__global__ void fully_fused_projection_bwd_radegs_kernel(
        // fwd inputs
        const uint32_t C,
        const uint32_t N,
        const T *__restrict__ means,    // [N, 3]
        const T *__restrict__ quats,    // [N, 4] optional
        const T *__restrict__ scales,   // [N, 3] optional
        const T *__restrict__ opacities,// [N, 1] optional
        const T *__restrict__ viewmats, // [C, 4, 4]
        const T *__restrict__ Ks,       // [C, 3, 3]
        const int32_t image_width,
        const int32_t image_height,
//            const T eps2d,
//            const bool ortho,
        // fwd outputs
        const T *__restrict__ radii,   // [C, N]
        const T *__restrict__ conic_opacities,        // [C, N, 4]
//            const T *__restrict__ compensations, // [C, N] optional
        // grad outputs
        const T *__restrict__ v_means2d,       // [C, N, 2]
        const T *__restrict__ v_depths,        // [C, N]
        const T *__restrict__ v_conics,        // [C, N, 3]
        const T *__restrict__ v_camera_planes, // [C, N, 6]
        const T *__restrict__ v_ray_planes,    // [C, N, 2]
        const T *__restrict__ v_normals,       // [C, N, 3]
        const T *__restrict__ v_ts,       // [C, N]
//            const T *__restrict__ v_compensations, // [C, N] optional
        // grad inputs
        T *__restrict__ v_means,   // [N, 3]
        T *__restrict__ v_quats,   // [N, 4] optional
        T *__restrict__ v_scales,  // [N, 3] optional
        T *__restrict__ v_viewmats, // [C, 4, 4] optional
        T *__restrict__ v_opacities
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N || radii[idx] <= 0) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    Ks += cid * 9;
    opacities += gid;

    conic_opacities += idx * 4;
    v_means2d += idx * 2;
    v_depths += idx;
    v_conics += idx * 4;
    v_camera_planes += idx * 6;
    v_ray_planes += idx * 2;
    v_normals += idx * 3;
    v_ts += idx;
    v_opacities += idx;

    // vjp: compute the inverse of the 2d covariance
    mat2<T> covar2d_inv = mat2<T>(conic_opacities[0], conic_opacities[1], conic_opacities[1], conic_opacities[2]);
    mat2<T> v_covar2d_inv =
            mat2<T>(v_conics[0], v_conics[1] * .5f, v_conics[1] * .5f, v_conics[2]);
    mat2<T> v_covar2d(0.f);
    inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d);

//        if (v_compensations != nullptr) {
//            // vjp: compensation term
//            const T compensation = compensations[idx];
//            const T v_compensation = v_compensations[idx];
//            add_blur_vjp(
//                    eps2d, covar2d_inv, compensation, v_compensation, v_covar2d
//            );
//        }

    // transform Gaussian to camera space
    mat3<T> R = mat3<T>(
            viewmats[0],
            viewmats[4],
            viewmats[8], // 1st column
            viewmats[1],
            viewmats[5],
            viewmats[9], // 2nd column
            viewmats[2],
            viewmats[6],
            viewmats[10] // 3rd column
    );
    vec3<T> t = vec3<T>(viewmats[3], viewmats[7], viewmats[11]);

    mat3<T> covar;
    vec4<T> quat;
    vec3<T> scale;

    // compute from quaternions and scales
    quat = glm::make_vec4(quats + gid * 4);
    scale = glm::make_vec3(scales + gid * 3);
    quat_scale_to_covar_preci<T>(quat, scale, &covar, nullptr);

    vec3<T> mean_c;
    pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
    mat3<T> covar_c;
    covar_world_to_cam(R, covar, covar_c);

    // vjp: perspective projection
    T fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    mat3<T> v_covar_c(0.f);
    vec3<T> v_mean_c(0.f);

    persp_proj_vjp_rade<T>(
            mean_c,
            covar_c,
            fx,
            fy,
            cx,
            cy,
            image_width,
            image_height,
            viewmats,
            glm::make_vec4(conic_opacities),
            reinterpret_cast<const vec2<T> *>(v_camera_planes),
            glm::make_vec2(v_ray_planes),
            glm::make_vec3(v_normals),
            glm::make_vec4(v_conics),
            v_covar2d,
            glm::make_vec2(v_means2d),
            v_mean_c,
            v_covar_c,
            *v_opacities
    );

    // add contribution from v_depths
    v_mean_c.z += v_depths[0];

    T norm = sqrt(mean_c[0] * mean_c[0] + mean_c[1] * mean_c[1] + mean_c[2] * mean_c[2]);
    v_mean_c += v_ts[0] * mean_c / norm;

    // vjp: transform Gaussian covariance to camera space
    vec3<T> v_mean(0.f);
    mat3<T> v_covar(0.f);
    mat3<T> v_R(0.f);
    vec3<T> v_t(0.f);
    pos_world_to_cam_vjp(
            R, t, glm::make_vec3(means), v_mean_c, v_R, v_t, v_mean
    );
    covar_world_to_cam_vjp(R, covar, v_covar_c, v_R, v_covar);

    // #if __CUDA_ARCH__ >= 700
    // write out results with warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, gid);
    if (v_means != nullptr) {
        warpSum(v_mean, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_means += gid * 3;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) {
                gpuAtomicAdd(v_means + i, v_mean[i]);
            }
        }
    }

    // Directly output gradients w.r.t. the quaternion and scale
    mat3<T> rotmat = quat_to_rotmat<T>(quat);
    vec4<T> v_quat(0.f);
    vec3<T> v_scale(0.f);
    quat_scale_to_covar_vjp<T>(
            quat, scale, rotmat, v_covar, v_quat, v_scale
    );
    warpSum(v_quat, warp_group_g);
    warpSum(v_scale, warp_group_g);
    if (warp_group_g.thread_rank() == 0) {
        v_quats += gid * 4;
        v_scales += gid * 3;
        gpuAtomicAdd(v_quats, v_quat[0]);
        gpuAtomicAdd(v_quats + 1, v_quat[1]);
        gpuAtomicAdd(v_quats + 2, v_quat[2]);
        gpuAtomicAdd(v_quats + 3, v_quat[3]);
        gpuAtomicAdd(v_scales, v_scale[0]);
        gpuAtomicAdd(v_scales + 1, v_scale[1]);
        gpuAtomicAdd(v_scales + 2, v_scale[2]);
    }

    if (v_viewmats != nullptr) {
        auto warp_group_c = cg::labeled_partition(warp, cid);
        warpSum(v_R, warp_group_c);
        warpSum(v_t, warp_group_c);
        if (warp_group_c.thread_rank() == 0) {
            v_viewmats += cid * 16;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) { // rows
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t j = 0; j < 3; j++) { // cols
                    gpuAtomicAdd(v_viewmats + i * 4 + j, v_R[j][i]);
                }
                gpuAtomicAdd(v_viewmats + i * 4 + 3, v_t[i]);
            }
        }
    }
}

std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
fully_fused_projection_bwd_radegs_tensor(
        // fwd inputs
        const torch::Tensor &means,                // [N, 3]
        const at::optional<torch::Tensor> &quats,  // [N, 4] optional
        const at::optional<torch::Tensor> &scales, // [N, 3] optional
        const torch::Tensor &opacities,                // [N, 1]
        const torch::Tensor &viewmats,             // [C, 4, 4]
        const torch::Tensor &Ks,                   // [C, 3, 3]
        const uint32_t image_width,
        const uint32_t image_height,
//            const float eps2d,
//            const bool ortho,
        // fwd outputs
        const torch::Tensor &radii,                       // [C, N]
        const torch::Tensor &conic_opacities,                      // [C, N, 4]
//            const at::optional<torch::Tensor> &compensations, // [C, N] optional
        // grad outputs
        const torch::Tensor &v_means2d,                     // [C, N, 2]
        const torch::Tensor &v_depths,                      // [C, N]
        const torch::Tensor &v_conics,                      // [C, N, 3]
        const torch::Tensor &v_camera_planes,               // [C, N, 6]
        const torch::Tensor &v_ray_planes,                  // [C, N, 2]
        const torch::Tensor &v_normals,                     // [C, N, 3]
        const torch::Tensor &v_ts,                     // [C, N]
//            const at::optional<torch::Tensor> &v_compensations, // [C, N] optional
        const bool viewmats_requires_grad
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
//        if (covars.has_value()) {
//            GSPLAT_CHECK_INPUT(covars.value());
//        } else {
        assert(quats.has_value() && scales.has_value());
        GSPLAT_CHECK_INPUT(quats.value());
        GSPLAT_CHECK_INPUT(scales.value());
//        }
    GSPLAT_CHECK_INPUT(viewmats);
    GSPLAT_CHECK_INPUT(Ks);
    GSPLAT_CHECK_INPUT(radii);
    GSPLAT_CHECK_INPUT(conic_opacities);
    GSPLAT_CHECK_INPUT(v_means2d);
    GSPLAT_CHECK_INPUT(v_depths);
    GSPLAT_CHECK_INPUT(v_conics);
    GSPLAT_CHECK_INPUT(v_camera_planes);
    GSPLAT_CHECK_INPUT(v_ray_planes);
    GSPLAT_CHECK_INPUT(v_normals);
    GSPLAT_CHECK_INPUT(v_ts);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor v_means = torch::zeros_like(means);
    torch::Tensor v_quats, v_scales; // optional
    v_quats = torch::zeros_like(quats.value());
    v_scales = torch::zeros_like(scales.value());
    torch::Tensor v_opacities = torch::zeros_like(scales.value());
//        }
    torch::Tensor v_viewmats;
    if (viewmats_requires_grad) {
        v_viewmats = torch::zeros_like(viewmats);
    }

    if (C && N) {
        if (means.scalar_type() == torch::kFloat32) {
            using T = float;
            fully_fused_projection_bwd_radegs_kernel<T>
            <<<(C * N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
            GSPLAT_N_THREADS,
            0,
            stream>>>(
                    C,
                    N,
                    means.data_ptr<T>(),
                    //                    covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
                    quats.value().data_ptr<T>(),
                    scales.value().data_ptr<T>(),
                    opacities.data_ptr<T>(),
                    viewmats.data_ptr<T>(),
                    Ks.data_ptr<T>(),
                    image_width,
                    image_height,
                    //                    eps2d,
                    //                    ortho,
                    radii.data_ptr<T>(),
                    conic_opacities.data_ptr<T>(),
                    //                    compensations.has_value()
                    //                    ? compensations.value().data_ptr<float>()
                    //                    : nullptr,
                    v_means2d.data_ptr<T>(),
                    v_depths.data_ptr<T>(),
                    v_conics.data_ptr<T>(),
                    v_camera_planes.data_ptr<T>(),
                    v_ray_planes.data_ptr<T>(),
                    v_normals.data_ptr<T>(),
                    v_ts.data_ptr<T>(),
                    //                    v_compensations.has_value()
                    //                    ? v_compensations.value().data_ptr<float>()
                    //                    : nullptr,
                    v_means.data_ptr<T>(),
                    //                    covars.has_value() ? v_covars.data_ptr<float>() : nullptr,
                    v_quats.data_ptr<T>(),
                    v_scales.data_ptr<T>(),
                    viewmats_requires_grad ? v_viewmats.data_ptr<T>() : nullptr,
                    v_opacities.data_ptr<T>()
            );
        } else if (means.scalar_type() == torch::kFloat64) {
            using T = double;
           // printf("  v_normals (fully fused tensor ): %f %f %f\n", v_normals[0].item<T>(), v_normals[1].item<T>(), v_normals[2].item<T>());

            fully_fused_projection_bwd_radegs_kernel<T>
            <<<(C * N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
            GSPLAT_N_THREADS,
            0,
            stream>>>(
                    C,
                    N,
                    means.data_ptr<T>(),
                    //                    covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
                    quats.value().data_ptr<T>(),
                    scales.value().data_ptr<T>(),
                    opacities.data_ptr<T>(),
                    viewmats.data_ptr<T>(),
                    Ks.data_ptr<T>(),
                    image_width,
                    image_height,
                    //                    eps2d,
                    //                    ortho,
                    radii.data_ptr<T>(),
                    conic_opacities.data_ptr<T>(),
                    //                    compensations.has_value()
                    //                    ? compensations.value().data_ptr<float>()
                    //                    : nullptr,
                    v_means2d.data_ptr<T>(),
                    v_depths.data_ptr<T>(),
                    v_conics.data_ptr<T>(),
                    v_camera_planes.data_ptr<T>(),
                    v_ray_planes.data_ptr<T>(),
                    v_normals.data_ptr<T>(),
                    v_ts.data_ptr<T>(),
                    //                    v_compensations.has_value()
                    //                    ? v_compensations.value().data_ptr<float>()
                    //                    : nullptr,
                    v_means.data_ptr<T>(),
                    //                    covars.has_value() ? v_covars.data_ptr<float>() : nullptr,
                    v_quats.data_ptr<T>(),
                    v_scales.data_ptr<T>(),
                    viewmats_requires_grad ? v_viewmats.data_ptr<T>() : nullptr,
                    v_opacities.data_ptr<T>()
            );
        } else {
            throw std::runtime_error("Unsupported scalar type");
        }
    }
    return std::make_tuple(v_means, v_quats, v_scales, v_viewmats);
}

} // namespace gsplat
