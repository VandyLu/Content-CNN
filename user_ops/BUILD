load("//tensorflow:tensorflow.bzl","tf_custom_op_library")

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)
tf_custom_op_library(
    name = "shift_corr.so",
    srcs = ["shift_corr.cc","shift_corr_kernel.cc"],
)
