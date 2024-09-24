//#include "bindings.h"
//#include <filesystem>
//
//void test_projection() {
//    // Call the function from the bindings
//    torch::Tensor means = torch::tensor({{-0.2567, -0.0691, -0.3114}}, torch::device(torch::kCUDA));
//    torch::Tensor quats = torch::tensor({{0.9390, -0.0000,  0.3435,  0.0189}}, torch::device(torch::kCUDA));
//    torch::Tensor scales = torch::tensor({{6.0428e-05, 3.0214e-03, 3.0214e-03}}, torch::device(torch::kCUDA));
//    torch::Tensor viewmats = torch::tensor({{-0.6671, -0.0924, -0.7393,  0.0243}}, torch::device(torch::kCUDA));
//    torch::Tensor Ks = torch::tensor({{{629.5340,   0.0000, 382.5000},
//         {  0.0000, 629.5340, 286.5000},
//         {  0.0000,   0.0000,   1.0000}}}, torch::device(torch::kCUDA));
//
//
//    int width = 764;
//    int height = 572;
//
//    float eps2d = 0.3;
//    bool packed = false;
//    float near_plane = 0.01;
//    float far_plane = 1000000.0;
//
//    const auto &[radii, means2d, depths,
//                conics, compensations, camera_plane,
//                output_normal, ray_plane, coef,
//                invraycov3Ds, ts] =
//        gsplat::fully_fused_projection_fwd_radegs_tensor(means, quats, scales, viewmats, Ks,
//                                                         width, height,
//                                                         eps2d, near_plane, far_plane, 0,
//                                                         false, false);
//
//
//
//}
//
//void test_rasterization() {
//
//    int width = 764;
//    int height = 572;
//    int tile_size = 16;
//
//    torch::Tensor means2d = torch::tensor(
//       {{{667.0242, 273.2691},
//         {514.8779, 474.9144},
//         {281.5622, 535.8053}}},
//     torch::device(torch::kCUDA));
//
//    torch::Tensor conics = torch::tensor(
//            {{{ 0.2418, -0.1072,  0.5018},
//         { 0.1069, -0.0159,  0.1608},
//         { 0.1206,  0.0142,  0.1271}}},
//        torch::device(torch::kCUDA));
//
//
//    torch::Tensor colors = torch::tensor(
//            {{{0.8314, 0.7725, 0.7294, 0.9560},
//         {0.7020, 0.7373, 0.7490, 0.7154},
//         {0.7608, 0.7216, 0.7137, 0.6633}}},
//         torch::device(torch::kCUDA));
//
//
//    torch::Tensor opacities = torch::tensor({0.1000, 0.1000, 0.1000}, torch::device(torch::kCUDA));
//
//    torch::Tensor camera_planes = torch::tensor(
//        {{{-1.3599e-03, -4.6963e-05, -4.0008e-05},
//         {-1.4453e-03,  3.9254e-03, -1.7806e-03},
//         {-1.3672e-03, -6.9270e-05, -3.7184e-05}}},
//            torch::device(torch::kCUDA));
//
//    torch::Tensor ray_planes = torch::tensor(
//            {{{ 0.0039, -0.0018},
//         { 0.0036, -0.0026},
//         { 0.0043, -0.0021}}},
//         torch::device(torch::kCUDA));
//
//    torch::Tensor ts = torch::tensor(
//            {{{1.0493, 0.7618, 0.7213}}},
//            torch::device(torch::kCUDA));
//
//    torch::Tensor normals = torch::tensor(
//        {{{-4.0569e-05, -1.4429e-03,  3.9805e-03},
//         {-2.0220e-03, -1.3533e-03, -7.7664e-05},
//         {-4.2565e-05, -1.4334e-03,  4.1763e-03}}},
//         torch::device(torch::kCUDA));
//
//    std::filesystem::path path = "/home/paul/vision/nerfstudio/isect_offsets.pth";
//    torch::Tensor tile_offsets;
//    torch::load(tile_offsets, path);
//
//    path = "/home/paul/vision/nerfstudio/flatten_ids.pth";
//    torch::Tensor flatten_ids;
//    torch::load(flatten_ids, path);
//
//
//
//    gsplat::rasterize_to_pixels_fwd_radegs_tensor(
//        // Gaussian parameters
//        means2d,   // [C, N, 2] or [nnz, 2]
//        conics,    // [C, N, 3] or [nnz, 3]
//        colors,    // [C, N, channels] or [nnz, channels]
//        opacities, // [C, N]  or [nnz]
//        camera_planes,
//        ray_planes,
//        ts,
//        normals,
//        at::nullopt,
//        at::nullopt,
//        width,
//        height,
//        tile_size,
//        // intersections
//        tile_offsets, // [C, tile_height, tile_width]
//        flatten_ids   // [n_isects]
//    ) ;
//
//}
//
//int main() {
//    test_projection();
//    test_rasterization();
//    return 0;
//}
