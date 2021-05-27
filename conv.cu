#include "conv.h"
#include "SE.h"
#include "downsample.h"

struct NetParameters{
    torch::Tensor Filter1;
    // Buffer for results
    torch::Tensor ConvOut1;
    torch::Tensor Filter2;
    torch::Tensor ConvOut2;
    torch::Tensor fixup_bias1b;
    torch::Tensor fixup_bias2a;
    torch::Tensor fixup_bias2b;
    torch::Tensor fixup_scale;
    torch::Tensor SE_W1;
    torch::Tensor SE_b1;
    torch::Tensor SE_W2;
    torch::Tensor SE_b2;
    torch::Tensor next_fixup_bias1a;

    // The downsampeld result will be multiplied to the SE block result and becomes
    // the activation of next layer. As a result, we will use the next activation
    // as a buffer to store downsample results
    // Last block, no next_fixup_bias1a
    NetParameters(torch::Tensor Filter1,
                  torch::Tensor ConvOut1,
                  torch::Tensor Filter2,
                  torch::Tensor ConvOut2,
                  torch::Tensor fixup_bias1b,
                  torch::Tensor fixup_bias2a,
                  torch::Tensor fixup_bias2b,
                  torch::Tensor SE_W1,
                  torch::Tensor SE_b1,
                  torch::Tensor SE_W2,
                  torch::Tensor SE_b2,
                  torch::Tensor fixup_scale)
        : Filter1(Filter1),
          ConvOut1(ConvOut1),
          Filter2(Filter2),
          ConvOut2(ConvOut2),
          fixup_bias1b(fixup_bias1b),
          fixup_bias2a(fixup_bias2a),
          fixup_bias2b(fixup_bias2b),
          SE_W1(SE_W1),
          SE_b1(SE_b1),
          SE_W2(SE_W2),
          SE_b2(SE_b2),
          fixup_scale(fixup_scale) {}

    // Blocks in the middle
    NetParameters(torch::Tensor Filter1,
                  torch::Tensor ConvOut1,
                  torch::Tensor Filter2,
                  torch::Tensor ConvOut2,
                  torch::Tensor fixup_bias1b,
                  torch::Tensor fixup_bias2a,
                  torch::Tensor fixup_bias2b,
                  torch::Tensor SE_W1,
                  torch::Tensor SE_b1,
                  torch::Tensor SE_W2,
                  torch::Tensor SE_b2,
                  torch::Tensor fixup_scale,
                  torch::Tensor next_fixup_bias1a)
        : Filter1(Filter1),
          ConvOut1(ConvOut1),
          Filter2(Filter2),
          ConvOut2(ConvOut2),
          fixup_bias1b(fixup_bias1b),
          fixup_bias2a(fixup_bias2a),
          fixup_bias2b(fixup_bias2b),
          SE_W1(SE_W1),
          SE_b1(SE_b1),
          SE_W2(SE_W2),
          SE_b2(SE_b2),
          fixup_scale(fixup_scale),
          next_fixup_bias1a(next_fixup_bias1a) {}
};

void net_after_stem(torch::Tensor Input,
                    NetParameters& Param1,
                    NetParameters& Param2,
                    NetParameters& Param3,
                    NetParameters& Param4) {
    ConvBlock<1, ImplicitGemm11, ImplicitGemm12>(
        Input, Param1.Filter1, Param1.ConvOut1, Param1.Filter2, Param1.ConvOut2,
        Param1.fixup_bias1b, Param1.fixup_bias2a, Param1.fixup_bias2b,
        Param1.fixup_scale);
    // First block does not have downsample, Param.Activation should be the same
    // as the input
    SE<4>(Param1.ConvOut2, Param1.SE_W1, Param1.SE_b1, Param1.SE_W2,
          Param1.SE_b2, Param1.next_fixup_bias1a, Input);
    // Downsample can potentially be done in parallel with ConvBlock. Maybe use two streams?

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // std::string name = std::string("Conv128x16x16x64NHWC3x3x64NHWC");
    // py::class_<Conv128x16x16x64NHWC3x3x64NHWC>(m, name.c_str())
    //     .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor,
    //                   torch::Tensor, torch::Tensor, torch::Tensor,
    //                   torch::Tensor, torch::Tensor>())
    //     .def("run", &Conv128x16x16x64NHWC3x3x64NHWC::run);

    // py::class_<MyCudaSE<8>>(m, "MyCudaSE")
    //     .def(
    //         py::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    //                  torch::Tensor, torch::Tensor>())
    //     .def("run", &MyCudaSE<8>::run);

    // py::class_<MyCudaSE2<16>>(m, "MyCudaSE2")
    //     .def(
    //         py::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    //                  torch::Tensor, torch::Tensor>())
    //     .def("run", &MyCudaSE2<16>::run);
    m.def("Conv2Block1", &ConvBlock<1, ImplicitGemm11, ImplicitGemm12>, "Block 1 Conv operations");
    m.def("Conv2Block2", &ConvBlock<2, ImplicitGemm21, ImplicitGemm22>, "Block 2 Conv operations");
    m.def("Conv2Block3", &ConvBlock<3, ImplicitGemm31, ImplicitGemm32>, "Block 3 Conv operations");
    m.def("Conv2Block4", &ConvBlock<4, ImplicitGemm41, ImplicitGemm42>, "Block 4 Conv operations");
    m.def("SE1", &SE<4>, "Block 1 SE operations");
    m.def("SE2", &SE<8>, "Block 2 SE operations");
    m.def("SE3", &SE_Tensor_Core<16>, "Block 3 SE operations");
    m.def("SE4", &SE_Tensor_Core<32>, "Block 4 SE operations");
    py::class_<NetParameters>(m, "NetParameters")
        .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor>())
        .def(
            py::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                     torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                     torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                     torch::Tensor>())
        .def_readwrite("Filter1", &NetParameters::Filter1)
        .def_readwrite("ConvOut1", &NetParameters::ConvOut1)
        .def_readwrite("Filter2", &NetParameters::Filter2)
        .def_readwrite("ConvOut2", &NetParameters::ConvOut2)
        .def_readwrite("fixup_bias1b", &NetParameters::fixup_bias1b)
        .def_readwrite("fixup_bias2a", &NetParameters::fixup_bias2a)
        .def_readwrite("fixup_bias2b", &NetParameters::fixup_bias2b)
        .def_readwrite("fixup_scale", &NetParameters::fixup_scale)
        .def_readwrite("SE_W1", &NetParameters::SE_W1)
        .def_readwrite("SE_b1", &NetParameters::SE_b1)
        .def_readwrite("SE_W2", &NetParameters::SE_W2)
        .def_readwrite("SE_b2", &NetParameters::SE_b2)
        .def_readwrite("next_fixup_bias1a", &NetParameters::next_fixup_bias1a);

    m.def("StemOp48_64", &StemOp48_64, "Stem Operation");
    



}