/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <time.h>
#include "tensorflow/core/example/feature_util.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow/cc/saved_model/loader.h"

#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

constexpr char kTestDataPbTxt[] =
    "cc/saved_model/testdata/half_plus_two_pbtxt/00000123";
constexpr char kTestDataMainOp[] =
    "cc/saved_model/testdata/half_plus_two_main_op/00000123";
constexpr char kTestDataSharded[] =
    //"cc/saved_model/testdata/half_plus_two/00000123";
    "cc/server/1";

class LoaderTest : public ::testing::Test {
 protected:
  LoaderTest() {}

  string MakeSerializedExample(float x) {
    //TODO: tf.train.features()
    tensorflow::Example example;
    auto* feature_map = example.mutable_features()->mutable_feature(); (*feature_map)["x"].mutable_float_list()->add_value(x);
    return example.SerializeAsString();
  }

  // clz: test input example
  string MakeSerializedExample_clz() {
    tensorflow::SequenceExample se;
    AppendFeatureValues({183}, GetFeatureList("timezone", &se) -> Add());
    AppendFeatureValues({11136}, GetFeatureList("creativeid", &se) -> Add());
    AppendFeatureValues({44064}, GetFeatureList("userid", &se) -> Add());
    AppendFeatureValues({0}, GetFeatureList("label", &se) -> Add());

//    tensorflow::Example example;
//    auto* feature_map = example.mutable_features()->mutable_feature();
//    (*feature_map)["x"].mutable_float_list()->add_value(x);
//return example.SerializeAsString();
    return se.SerializeAsString();
  }

  void ValidateAssets(const string& export_dir,
                      const SavedModelBundle& bundle) {
    const string asset_directory =
        io::JoinPath(export_dir, kSavedModelAssetsDirectory);
    const string asset_filename = "foo.txt";
    const string asset_filepath = io::JoinPath(asset_directory, asset_filename);
    TF_EXPECT_OK(Env::Default()->FileExists(asset_filepath));

    std::vector<Tensor> path_outputs;
    TF_ASSERT_OK(
        bundle.session->Run({}, {"filename_tensor:0"}, {}, &path_outputs));
    ASSERT_EQ(1, path_outputs.size());

    test::ExpectTensorEqual<string>(
        test::AsTensor<string>({"foo.txt"}, TensorShape({})), path_outputs[0]);
  }

  void CheckSavedModelBundle(const string& export_dir,
                             const SavedModelBundle& bundle) {
    // ValidateAssets(export_dir, bundle);
    // Retrieve the regression signature from meta graph def.
    clock_t start = clock();
    std::cout<<"[CheckSavedModelBundle]"<<std::endl;
    const auto signature_def_map = bundle.meta_graph_def.signature_def();
    const auto signature_def = signature_def_map.at("predict");

    const string input_name = signature_def.inputs().at("examples").name();
    const string output_name =
        signature_def.outputs().at("probabilities").name();

    std::cout<< "[CheckSavedModelBundle] input_name: " <<input_name <<std::endl;
    std::cout<< "[CheckSavedModelBundle] output_name: " <<output_name <<std::endl;

    std::vector<string> serialized_examples;
//    for (float x : {0, 1, 2, 3}) {
//      serialized_examples.push_back(MakeSerializedExample(x));
//    }
    serialized_examples.push_back(MakeSerializedExample_clz());
    clock_t mid = clock();
    std::cout<<"[CheckSavedModelBundle] check input"<<std::endl;;
    Tensor input =
        test::AsTensor<string>(serialized_examples, TensorShape({1}));
    std::vector<Tensor> outputs;
    bundle.session->Run({{input_name, input}}, {output_name}, {},
                                     &outputs);
    clock_t ends = clock();


//    // Validate the half plus two behavior.
//    Tensor input =
//        test::AsTensor<string>(serialized_examples, TensorShape({4}));
//    std::vector<Tensor> outputs;
//    TF_ASSERT_OK(bundle.session->Run({{input_name, input}}, {output_name}, {},
//                                     &outputs));
    std::cout << "output:"<< std::endl;
    for (int i = 0; i < 2; i++){
	    std::cout<< i  + ":"<< outputs[0].matrix<float>()(0,i)<< std::endl;
    }
    std::cout <<"Running mid Time : "<<(double)(mid - start)/ CLOCKS_PER_SEC << std::endl;
    std::cout <<"Running end Time : "<<(double)(ends - start)/ CLOCKS_PER_SEC << std::endl;
   ASSERT_EQ(outputs.size(), 1);
   test::ExpectTensorEqual<float>(
       outputs[0],
       test::AsTensor<float>({2, 2.5}, TensorShape({1, 2})));
  }
};

// Test for resource leaks related to TensorFlow session closing requirements
// when loading and unloading large numbers of SavedModelBundles.
// TODO(sukritiramesh): Increase run iterations and move outside of the test
// suite.
TEST_F(LoaderTest, TagMatch) {
  SavedModelBundle bundle;
  SessionOptions session_options;
  RunOptions run_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataSharded);
  TF_ASSERT_OK(LoadSavedModel(session_options, run_options, export_dir,
                              {kSavedModelTagServe}, &bundle));
  CheckSavedModelBundle(export_dir, bundle);
}


}  // namespace
}  // namespace tensorflow
