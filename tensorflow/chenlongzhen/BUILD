cc_library(
    name = "simple_servers",
    srcs = ["simple_servers.cc"],
    hdrs = ["simple_servers.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":saved_model_bundle_source_adapter",
        ":session_bundle_source_adapter_proto",
        "//tensorflow_serving/core:aspired_versions_manager_builder",
        "//tensorflow_serving/core:availability_preserving_policy",
        "//tensorflow_serving/core:loader",
        "//tensorflow_serving/core:manager",
        "//tensorflow_serving/core:source",
        "//tensorflow_serving/core:source_adapter",
        "//tensorflow_serving/core:storage_path",
        "//tensorflow_serving/core:target",
        "//tensorflow_serving/sources/storage_path:file_system_storage_path_source",
        "//tensorflow_serving/sources/storage_path:file_system_storage_path_source_proto",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "simple_servers_test",
    srcs = ["simple_servers_test.cc"],
    data = ["@org_tensorflow//tensorflow/cc/saved_model:saved_model_half_plus_two"],
    # Link in all registered kernels.
    linkstatic = 1,
    deps = [
        ":simple_servers",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:testlib",
    ],
)
