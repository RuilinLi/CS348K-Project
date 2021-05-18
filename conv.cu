#include "conv.h"
#include "SE.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    std::string name = std::string("Conv128x16x16x64NHWC3x3x64NHWC");
    py::class_<Conv128x16x16x64NHWC3x3x64NHWC>(m, name.c_str())
        .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor,
                      torch::Tensor, torch::Tensor>())
        .def("run", &Conv128x16x16x64NHWC3x3x64NHWC::run);

    py::class_<MyCudaSE<8>>(m, "MyCudaSE")
        .def(
            py::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                     torch::Tensor, torch::Tensor>())
        .def("run", &MyCudaSE<8>::run);
    
    py::class_<ResnetSE>(m, "ResnetSE")
        .def(
            py::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                     torch::Tensor, torch::Tensor, torch::Tensor>())
        .def("run", &ResnetSE::reduce);
}