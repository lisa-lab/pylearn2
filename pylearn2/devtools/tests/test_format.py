"""
Unit tests for format checking
"""

from __future__ import print_function

from nose.plugins.skip import SkipTest

import os
import pylearn2

from pylearn2.devtools.tests.docscrape import docstring_errors
from pylearn2.devtools.list_files import list_files
from pylearn2.devtools.tests.pep8.pep8 import StyleGuide

whitelist_pep8 = [
    "rbm_tools.py",
    "distributions/mnd.py",
    "models/sparse_autoencoder.py",
    "models/tests/test_dbm.py",
    "models/tests/test_s3c_inference.py",
    "models/tests/test_mnd.py",
    "models/tests/test_s3c_misc.py",
    "models/gsn.py",
    "models/differentiable_sparse_coding.py",
    "models/local_coordinate_coding.py",
    "models/mnd.py",
    "models/s3c.py",
    "tests/test_monitor.py",
    "kmeans.py",
    "packaged_dependencies/theano_linear/conv2d.py",
    "packaged_dependencies/theano_linear/imaging.py",
    "packaged_dependencies/theano_linear/pyramid.py",
    "packaged_dependencies/theano_linear/unshared_conv/"
    "test_gpu_unshared_conv.py",
    "packaged_dependencies/theano_linear/unshared_conv/"
    "test_localdot.py",
    "packaged_dependencies/theano_linear/unshared_conv/localdot.py",
    "packaged_dependencies/theano_linear/unshared_conv/"
    "unshared_conv.py",
    "packaged_dependencies/theano_linear/linear.py",
    "packaged_dependencies/theano_linear/test_spconv.py",
    "packaged_dependencies/theano_linear/test_matrixmul.py",
    "packaged_dependencies/theano_linear/spconv.py",
    "expr/tests/test_coding.py",
    "expr/tests/test_normalize.py",
    "expr/tests/test_stochastic_pool.py",
    "expr/stochastic_pool.py",
    "expr/sampling.py",
    "expr/information_theory.py",
    "expr/basic.py",
    "gui/graph_2D.py",
    "sandbox/cuda_convnet/weight_acts.py",
    "sandbox/cuda_convnet/filter_acts.py",
    "sandbox/cuda_convnet/tests/test_filter_acts_strided.py",
    "sandbox/cuda_convnet/tests/test_probabilistic_max_pooling.py",
    "sandbox/cuda_convnet/tests/test_filter_acts.py",
    "sandbox/cuda_convnet/tests/test_weight_acts_strided.py",
    "sandbox/cuda_convnet/tests/test_image_acts_strided.py",
    "sandbox/cuda_convnet/tests/test_img_acts.py",
    "sandbox/cuda_convnet/tests/test_stochastic_pool.py",
    "sandbox/cuda_convnet/specialized_bench.py",
    "sandbox/cuda_convnet/response_norm.py",
    "sandbox/cuda_convnet/__init__.py",
    "sandbox/cuda_convnet/img_acts.py",
    "sandbox/cuda_convnet/convnet_compile.py",
    "sandbox/cuda_convnet/pthreads.py",
    "sandbox/cuda_convnet/pool.py",
    "sandbox/cuda_convnet/bench.py",
    "sandbox/cuda_convnet/stochastic_pool.py",
    "sandbox/cuda_convnet/probabilistic_max_pooling.py",
    "sandbox/tuple_var.py",
    "sandbox/lisa_rl/bandit/average_agent.py",
    "sandbox/lisa_rl/bandit/classifier_bandit.py",
    "sandbox/lisa_rl/bandit/classifier_agent.py",
    "sandbox/lisa_rl/bandit/plot_reward.py",
    "config/old_config.py",
    "utils/utlc.py",
    "utils/tests/test_serial.py",
    "utils/common_strings.py",
    "utils/mem.py",
    "dataset_get/dataset-get.py",
    "dataset_get/helper-scripts/make-archive.py",
    "dataset_get/dataset_resolver.py",
    "optimization/minres.py",
    "linear/conv2d.py",
    "linear/local_c01b.py",
    "linear/linear_transform.py",
    "linear/conv2d_c01b.py",
    "energy_functions/rbm_energy.py",
    "scripts/pkl_inspector.py",
    "scripts/show_binocular_greyscale_examples.py",
    "scripts/jobman/tester.py",
    "scripts/papers/maxout/svhn_preprocessing.py",
    "scripts/papers/jia_huang_wkshp_11/fit_final_model.py",
    "scripts/papers/jia_huang_wkshp_11/evaluate.py",
    "scripts/papers/jia_huang_wkshp_11/extract_features.py",
    "scripts/papers/jia_huang_wkshp_11/assemble.py",
    "scripts/gpu_pkl_to_cpu_pkl.py",
    "scripts/gsn_example.py",
    "scripts/tutorials/deep_trainer/run_deep_trainer.py",
    "scripts/tutorials/grbm_smd/test_grbm_smd.py",
    "scripts/icml_2013_wrepl/multimodal/"
    "extract_layer_2_kmeans_features.py",
    "scripts/icml_2013_wrepl/multimodal/make_submission.py",
    "scripts/icml_2013_wrepl/multimodal/lcn.py",
    "scripts/icml_2013_wrepl/multimodal/extract_kmeans_features.py",
    "scripts/icml_2013_wrepl/emotions/emotions_dataset.py",
    "scripts/icml_2013_wrepl/emotions/make_submission.py",
    "scripts/icml_2013_wrepl/black_box/black_box_dataset.py",
    "scripts/icml_2013_wrepl/black_box/make_submission.py",
    "scripts/diff_monitor.py",
    "corruption.py",
    "sandbox/lisa_rl/bandit/gaussian_bandit.py",
    "utils/track_version.py",
    "scripts/get_version.py",
    "training_algorithms/tests/test_bgd.py",
    "training_algorithms/tests/test_default.py",
    "training_algorithms/default.py",
    "training_algorithms/training_algorithm.py",
    "distributions/tests/test_mnd.py",
    "distributions/parzen.py",
    "distributions/uniform_hypersphere.py",
    "models/setup.py",
    "models/independent_multiclass_logistic.py",
    "models/softmax_regression.py",
    "models/tests/test_reflection_clip.py",
    "models/tests/test_maxout.py",
    "models/tests/test_convelemwise_sigm.py",
    "models/dbm/sampling_procedure.py",
    "models/rbm.py",
    "models/pca.py",
    "tests/test_train.py",
    "packaged_dependencies/theano_linear/unshared_conv/gpu_unshared_conv.py",
    "packaged_dependencies/theano_linear/unshared_conv/test_unshared_conv.py",
    "packaged_dependencies/theano_linear/linearmixin.py",
    "packaged_dependencies/theano_linear/util.py",
    "packaged_dependencies/theano_linear/__init__.py",
    "packaged_dependencies/theano_linear/test_linear.py",
    "expr/tests/test_nnet.py",
    "expr/image.py",
    "expr/coding.py",
    "expr/normalize.py",
    "expr/probabilistic_max_pooling.py",
    "testing/tests/test.py",
    "testing/skip.py",
    "testing/prereqs.py",
    "testing/__init__.py",
    "gui/get_weights_report.py",
    "gui/patch_viewer.py",
    "sandbox/cuda_convnet/tests/test_response_norm.py",
    "sandbox/cuda_convnet/tests/profile_probabilistic_max_pooling.py",
    "sandbox/cuda_convnet/tests/test_rop_pool.py",
    "sandbox/cuda_convnet/tests/test_pool.py",
    "sandbox/cuda_convnet/tests/test_common.py",
    "sandbox/cuda_convnet/shared_code.py",
    "sandbox/cuda_convnet/code_templates.py",
    "sandbox/lisa_rl/bandit/agent.py",
    "sandbox/lisa_rl/bandit/algorithm.py",
    "sandbox/lisa_rl/bandit/environment.py",
    "sandbox/lisa_rl/__init__.py",
    "datasets/avicenna.py",
    "datasets/iris.py",
    "datasets/adult.py",
    "datasets/npy_npz.py",
    "datasets/control.py",
    "datasets/cifar100.py",
    "datasets/transformer_dataset.py",
    "termination_criteria/__init__.py",
    "__init__.py",
    "utils/logger.py",
    "utils/tests/test_mnist_ubyte.py",
    "utils/tests/test_data_specs.py",
    "utils/tests/test_bit_strings.py",
    "utils/tests/test_iteration.py",
    "utils/theano_graph.py",
    "utils/__init__.py",
    "utils/datasets.py",
    "utils/data_specs.py",
    "utils/insert_along_axis.py",
    "utils/environ.py",
    "utils/call_check.py",
    "utils/python26.py",
    "deprecated/classifier.py",
    "train.py",
    "classifier.py",
    "dataset_get/helper-scripts/make-sources.py",
    "pca.py",
    "optimization/test_linesearch.py",
    "optimization/test_minres.py",
    "optimization/test_batch_gradient_descent.py",
    "optimization/linear_cg.py",
    "optimization/test_feature_sign.py",
    "optimization/feature_sign.py",
    "optimization/test_linear_cg.py",
    "optimization/linesearch.py",
    "linear/tests/test_conv2d.py",
    "linear/tests/test_conv2d_c01b.py",
    "linear/matrixmul.py",
    "energy_functions/energy_function.py",
    "scripts/make_weights_image.py",
    "scripts/plot_monitor.py",
    "scripts/print_monitor.py",
    "scripts/num_parameters.py",
    "scripts/benchmark/time_relu.py",
    "scripts/jobman/experiment.py",
    "scripts/jobman/__init__.py",
    "scripts/dbm/show_negative_chains.py",
    "scripts/papers/maxout/compute_test_err.py",
    "scripts/papers/jia_huang_wkshp_11/npy2mat.py",
    "scripts/datasets/step_through_small_norb.py",
    "scripts/datasets/step_through_norb_foveated.py",
    "scripts/datasets/make_downsampled_stl10.py",
    "scripts/datasets/browse_small_norb.py",
    "scripts/datasets/make_mnistplus.py",
    "scripts/mlp/predict_csv.py",
    "scripts/find_gpu_fields.py",
    "scripts/tutorials/deep_trainer/test_deep_trainer.py",
    "scripts/icml_2013_wrepl/multimodal/make_wordlist.py",
    "base.py",
    "devtools/tests/test_via_pyflakes.py",
    "devtools/tests/test_shebangs.py",
    "devtools/tests/pep8/pep8.py",
    "devtools/tests/docscrape.py",
    "devtools/run_pyflakes.py",
    "devtools/record.py",
    "train_extensions/tests/test_window_flip.py",
    "train_extensions/__init__.py",
]

whitelist_docstrings = [
    'scripts/datasets/step_through_norb_foveated.py',
    'blocks.py',
    'datasets/hdf5.py',
    'rbm_tools.py',
    'training_algorithms/tests/test_bgd.py',
    'training_algorithms/tests/test_sgd.py',
    'training_algorithms/tests/test_default.py',
    'training_algorithms/bgd.py',
    'training_algorithms/default.py',
    'training_algorithms/training_algorithm.py',
    'training_algorithms/__init__.py',
    'training_algorithms/sgd.py',
    'distributions/tests/test_mnd.py',
    'distributions/multinomial.py',
    'distributions/parzen.py',
    'distributions/__init__.py',
    'distributions/mnd.py',
    'distributions/uniform_hypersphere.py',
    'models/setup.py',
    'models/independent_multiclass_logistic.py',
    'models/softmax_regression.py',
    'models/sparse_autoencoder.py',
    'models/tests/test_reflection_clip.py',
    'models/tests/test_dbm.py',
    'models/tests/test_gsn.py',
    'models/tests/test_dropout.py',
    'models/tests/test_autoencoder.py',
    'models/tests/test_mlp.py',
    'models/tests/test_s3c_inference.py',
    'models/tests/test_maxout.py',
    'models/tests/test_mnd.py',
    'models/tests/test_vae.py',
    'models/tests/test_rbm.py',
    'models/tests/test_s3c_misc.py',
    'models/gsn.py',
    'models/dbm/sampling_procedure.py',
    'models/differentiable_sparse_coding.py',
    'models/local_coordinate_coding.py',
    'models/maxout.py',
    'models/s3c.py',
    'models/mnd.py',
    'models/rbm.py',
    'models/autoencoder.py',
    'tests/test_dbm_metrics.py',
    'tests/test_monitor.py',
    'tests/test_train.py',
    'tests/rbm/test_ais.py',
    'kmeans.py',
    'packaged_dependencies/__init__.py',
    'packaged_dependencies/theano_linear/imaging.py',
    'packaged_dependencies/theano_linear/unshared_conv/__init__.py',
    'packaged_dependencies/theano_linear/unshared_conv/unshared_conv.py',
    'packaged_dependencies/theano_linear/linearmixin.py',
    'packaged_dependencies/theano_linear/linear.py',
    'packaged_dependencies/theano_linear/test_spconv.py',
    'expr/activations.py',
    'expr/tests/test_probabilistic_max_pooling.py',
    'expr/tests/test_preprocessing.py',
    'expr/tests/test_nnet.py',
    'expr/tests/test_coding.py',
    'expr/tests/test_normalize.py',
    'expr/tests/test_stochastic_pool.py',
    'expr/preprocessing.py',
    'expr/image.py',
    'expr/coding.py',
    'expr/__init__.py',
    'expr/stochastic_pool.py',
    'expr/sampling.py',
    'expr/normalize.py',
    'expr/probabilistic_max_pooling.py',
    'expr/information_theory.py',
    'expr/basic.py',
    'testing/tests/test.py',
    'testing/skip.py',
    'testing/prereqs.py',
    'testing/__init__.py',
    'testing/datasets.py',
    'gui/get_weights_report.py',
    'gui/__init__.py',
    'gui/patch_viewer.py',
    'scalar.py',
    'sandbox/cuda_convnet/weight_acts.py',
    'sandbox/cuda_convnet/filter_acts.py',
    'sandbox/cuda_convnet/tests/test_filter_acts_strided.py',
    'sandbox/cuda_convnet/tests/test_probabilistic_max_pooling.py',
    'sandbox/cuda_convnet/tests/test_filter_acts.py',
    'sandbox/cuda_convnet/tests/test_img_acts.py',
    'sandbox/cuda_convnet/tests/test_response_norm.py',
    'sandbox/cuda_convnet/tests/profile_probabilistic_max_pooling.py',
    'sandbox/cuda_convnet/tests/test_weight_acts.py',
    'sandbox/cuda_convnet/tests/test_rop_pool.py',
    'sandbox/cuda_convnet/tests/test_pool.py',
    'sandbox/cuda_convnet/tests/test_common.py',
    'sandbox/cuda_convnet/tests/test_stochastic_pool.py',
    'sandbox/cuda_convnet/shared_code.py',
    'sandbox/cuda_convnet/__init__.py',
    'sandbox/cuda_convnet/img_acts.py',
    'sandbox/cuda_convnet/base_acts.py',
    'sandbox/cuda_convnet/pool.py',
    'sandbox/cuda_convnet/stochastic_pool.py',
    'sandbox/cuda_convnet/code_templates.py',
    'sandbox/cuda_convnet/probabilistic_max_pooling.py',
    'sandbox/tuple_var.py',
    'sandbox/__init__.py',
    'sandbox/lisa_rl/bandit/simulator.py',
    'sandbox/lisa_rl/bandit/agent.py',
    'sandbox/lisa_rl/bandit/algorithm.py',
    'sandbox/lisa_rl/bandit/environment.py',
    'sandbox/lisa_rl/bandit/average_agent.py',
    'sandbox/lisa_rl/bandit/classifier_bandit.py',
    'sandbox/lisa_rl/bandit/__init__.py',
    'sandbox/lisa_rl/bandit/classifier_agent.py',
    'sandbox/lisa_rl/bandit/gaussian_bandit.py',
    'sandbox/lisa_rl/__init__.py',
    'config/old_config.py',
    'config/tests/test_yaml_parse.py',
    'config/yaml_parse.py',
    'space/tests/test_space.py',
    'space/__init__.py',
    'datasets/norb.py',
    'datasets/utlc.py',
    'datasets/mnistplus.py',
    'datasets/cos_dataset.py',
    'datasets/cifar10.py',
    'datasets/svhn.py',
    'datasets/tests/test_preprocessing.py',
    'datasets/tests/test_mnist.py',
    'datasets/tests/test_imports.py',
    'datasets/tests/test_cifar10.py',
    'datasets/tests/test_norb.py',
    'datasets/tests/test_dense_design_matrix.py',
    'datasets/tests/test_vector_spaces_dataset.py',
    'datasets/tests/test_four_regions.py',
    'datasets/tests/test_csv_dataset.py',
    'datasets/tests/test_icml07.py',
    'datasets/tests/test_utlc.py',
    'datasets/preprocessing.py',
    'datasets/avicenna.py',
    'datasets/iris.py',
    'datasets/config.py',
    'datasets/dense_design_matrix.py',
    'datasets/adult.py',
    'datasets/tfd.py',
    'datasets/icml07.py',
    'datasets/filetensor.py',
    'datasets/npy_npz.py',
    'datasets/hepatitis.py',
    'datasets/wiskott.py',
    'datasets/control.py',
    'datasets/exc.py',
    'datasets/__init__.py',
    'datasets/mnist.py',
    'datasets/sparse_dataset.py',
    'datasets/csv_dataset.py',
    'datasets/cifar100.py',
    'datasets/tl_challenge.py',
    'datasets/transformer_dataset.py',
    'datasets/norb_small.py',
    'datasets/retina.py',
    'datasets/ocr.py',
    'datasets/stl10.py',
    'datasets/matlab_dataset.py',
    'datasets/vector_spaces_dataset.py',
    'datasets/four_regions.py',
    'datasets/debug.py',
    'datasets/binarizer.py',
    'termination_criteria/__init__.py',
    '__init__.py',
    'utils/utlc.py',
    'utils/setup.py',
    'utils/compile.py',
    'utils/logger.py',
    'utils/general.py',
    'utils/testing.py',
    'utils/tests/test_mnist_ubyte.py',
    'utils/tests/test_data_specs.py',
    'utils/tests/test_video.py',
    'utils/tests/test_bit_strings.py',
    'utils/tests/test_rng.py',
    'utils/tests/test_pooling.py',
    'utils/tests/test_iteration.py',
    'utils/tests/test_insert_along_axis.py',
    'utils/tests/test_utlc.py',
    'utils/tests/test_compile.py',
    'utils/tests/test_key_aware.py',
    'utils/key_aware.py',
    'utils/video.py',
    'utils/bit_strings.py',
    'utils/iteration.py',
    'utils/pooling.py',
    'utils/theano_graph.py',
    'utils/common_strings.py',
    'utils/datasets.py',
    'utils/data_specs.py',
    'utils/shell.py',
    'utils/rng.py',
    'utils/insert_along_axis.py',
    'utils/environ.py',
    'utils/call_check.py',
    'utils/mnist_ubyte.py',
    'utils/track_version.py',
    'utils/mem.py',
    'utils/python26.py',
    'utils/timing.py',
    'deprecated/__init__.py',
    'deprecated/classifier.py',
    'train.py',
    'format/tests/test_target_format.py',
    'format/__init__.py',
    'dataset_get/dataset-get.py',
    'dataset_get/helper-scripts/make-sources.py',
    'dataset_get/helper-scripts/make-archive.py',
    'dataset_get/dataset_resolver.py',
    'pca.py',
    'monitor.py',
    'optimization/batch_gradient_descent.py',
    'optimization/__init__.py',
    'optimization/test_batch_gradient_descent.py',
    'optimization/linear_cg.py',
    'optimization/minres.py',
    'optimization/test_feature_sign.py',
    'optimization/feature_sign.py',
    'optimization/linesearch.py',
    'linear/conv2d.py',
    'linear/tests/test_matrixmul.py',
    'linear/local_c01b.py',
    'linear/matrixmul.py',
    'linear/__init__.py',
    'linear/linear_transform.py',
    'linear/conv2d_c01b.py',
    'energy_functions/tests/__init__.py',
    'energy_functions/rbm_energy.py',
    'energy_functions/__init__.py',
    'energy_functions/energy_function.py',
    'scripts/plot_monitor.py',
    'scripts/print_model.py',
    'scripts/tests/__init__.py',
    'scripts/pkl_inspector.py',
    'scripts/get_version.py',
    'scripts/print_monitor.py',
    'scripts/show_binocular_greyscale_examples.py',
    'scripts/num_parameters.py',
    'scripts/jobman/tester.py',
    'scripts/jobman/experiment.py',
    'scripts/jobman/__init__.py',
    'scripts/papers/__init__.py',
    'scripts/papers/jia_huang_wkshp_11/extract_features.py',
    'scripts/print_channel_doc.py',
    'scripts/gpu_pkl_to_cpu_pkl.py',
    'scripts/datasets/step_through_small_norb.py',
    'scripts/datasets/download_mnist.py',
    'scripts/datasets/download_binarized_mnist.py',
    'scripts/datasets/browse_small_norb.py',
    'scripts/datasets/make_mnistplus.py',
    'scripts/__init__.py',
    'scripts/gsn_example.py',
    'scripts/mlp/predict_csv.py',
    'scripts/mlp/__init__.py',
    'scripts/find_gpu_fields.py',
    'scripts/tutorials/dbm_demo/train_dbm.py',
    'scripts/tutorials/dbm_demo/__init__.py',
    'scripts/tutorials/tests/test_dbm.py',
    'scripts/tutorials/tests/test_mlp_nested.py',
    'scripts/tutorials/multilayer_perceptron/tests/test_mlp.py',
    'scripts/tutorials/softmax_regression/tests/test_softmaxreg.py',
    'scripts/tutorials/deep_trainer/__init__.py',
    'scripts/tutorials/deep_trainer/run_deep_trainer.py',
    'scripts/tutorials/grbm_smd/make_dataset.py',
    'scripts/tutorials/grbm_smd/__init__.py',
    'scripts/tutorials/grbm_smd/test_grbm_smd.py',
    'scripts/tutorials/__init__.py',
    'scripts/tutorials/jobman_demo/utils.py',
    'scripts/tutorials/jobman_demo/__init__.py',
    'scripts/tutorials/stacked_autoencoders/tests/test_dae.py',
    'scripts/icml_2013_wrepl/__init__.py',
    'scripts/icml_2013_wrepl/multimodal/extract_layer_2_kmeans_features.py',
    'scripts/icml_2013_wrepl/multimodal/make_submission.py',
    'scripts/icml_2013_wrepl/multimodal/lcn.py',
    'scripts/icml_2013_wrepl/multimodal/__init__.py',
    'scripts/icml_2013_wrepl/multimodal/extract_kmeans_features.py',
    'scripts/icml_2013_wrepl/emotions/emotions_dataset.py',
    'scripts/icml_2013_wrepl/emotions/make_submission.py',
    'scripts/icml_2013_wrepl/emotions/__init__.py',
    'scripts/icml_2013_wrepl/black_box/black_box_dataset.py',
    'scripts/icml_2013_wrepl/black_box/make_submission.py',
    'scripts/icml_2013_wrepl/black_box/__init__.py',
    'scripts/diff_monitor.py',
    'base.py',
    'devtools/tests/test_via_pyflakes.py',
    'devtools/tests/test_shebangs.py',
    'devtools/tests/__init__.py',
    'devtools/tests/docscrape.py',
    'devtools/run_pyflakes.py',
    'devtools/__init__.py',
    'devtools/record.py',
    'corruption.py',
    'datasets/tests/test_tl_challenge.py',
    'datasets/tests/test_tfd.py',
    'datasets/tests/test_npy_npz.py',
    'linear/tests/test_conv2d.py',
    'devtools/tests/pep8/pep8.py',
    'devtools/tests/pep8/__init__.py',
    'scripts/lcc_tangents/make_dataset.py',
    'scripts/icml_2013_wrepl/multimodal/make_wordlist.py',
    'scripts/datasets/make_stl10_whitened.py',
    'scripts/datasets/make_stl10_patches_8x8.py',
    'scripts/datasets/make_stl10_patches.py',
    'scripts/datasets/make_cifar10_whitened.py',
    'scripts/datasets/make_cifar10_gcn_whitened.py',
    'scripts/datasets/make_cifar100_patches.py',
    'scripts/datasets/make_cifar100_gcn_whitened.py',
    'scripts/datasets/make_svhn_pytables.py',
    'energy_functions/tests/test_rbm_energy.py',
]

# add files which fail to run to whitelist_docstrings
whitelist_docstrings.extend([
    'sandbox/rnn/models/mlp_hook.py',
    'training_algorithms/tests/test_learning_rule.py',
    'models/pca.py',
    'datasets/tests/test_hdf5.py',
    'linear/tests/test_conv2d_c01b.py',
    'packaged_dependencies/theano_linear/conv2d.py',
    'packaged_dependencies/theano_linear/pyramid.py',
    'packaged_dependencies/theano_linear/unshared_conv/gpu_unshared_conv.py',
    'packaged_dependencies/theano_linear/unshared_conv/'
    'test_gpu_unshared_conv.py',
    'packaged_dependencies/theano_linear/unshared_conv/test_localdot.py',
    'packaged_dependencies/theano_linear/unshared_conv/test_unshared_conv.py',
    'packaged_dependencies/theano_linear/unshared_conv/localdot.py',
    'packaged_dependencies/theano_linear/util.py',
    'packaged_dependencies/theano_linear/__init__.py',
    'packaged_dependencies/theano_linear/test_matrixmul.py',
    'packaged_dependencies/theano_linear/test_linear.py',
    'packaged_dependencies/theano_linear/spconv.py',
    'sandbox/cuda_convnet/tests/test_weight_acts_strided.py',
    'sandbox/cuda_convnet/tests/test_image_acts_strided.py',
    'sandbox/cuda_convnet/specialized_bench.py',
    'sandbox/cuda_convnet/response_norm.py',
    'sandbox/cuda_convnet/convnet_compile.py',
    'sandbox/cuda_convnet/pthreads.py',
    'sandbox/cuda_convnet/bench.py',
    'sandbox/lisa_rl/bandit/plot_reward.py',
    'sandbox/lisa_rl/bandit/simulate.py',
    'config/__init__.py',
    'utils/__init__.py',
    'optimization/test_linesearch.py',
    'optimization/test_minres.py',
    'optimization/test_linear_cg.py',
    'scripts/papers/maxout/svhn_preprocessing.py',
    'scripts/papers/maxout/compute_test_err.py',
    'scripts/papers/jia_huang_wkshp_11/fit_final_model.py',
    'scripts/papers/jia_huang_wkshp_11/evaluate.py',
    'scripts/papers/jia_huang_wkshp_11/npy2mat.py',
    'scripts/papers/jia_huang_wkshp_11/assemble.py',
    'scripts/datasets/make_cifar100_patches_8x8.py',
    'scripts/datasets/make_downsampled_stl10.py',
    'scripts/datasets/make_cifar100_whitened.py',
    'scripts/tutorials/deep_trainer/test_deep_trainer.py',
    'scripts/icml_2013_wrepl/black_box/learn_zca.py',
    'train_extensions/tests/test_window_flip.py',
    'train_extensions/window_flip.py',
    'linear/tests/test_local_c01b.py',
    'sandbox/cuda_convnet/debug.py', ])


def test_format_pep8():
    """
    Test if pep8 is respected.
    """
    pep8_checker = StyleGuide()
    files_to_check = []
    for path in list_files(".py"):
        rel_path = os.path.relpath(path, pylearn2.__path__[0])
        if rel_path in whitelist_pep8:
            continue
        else:
            files_to_check.append(path)
    report = pep8_checker.check_files(files_to_check)
    if report.total_errors > 0:
        raise AssertionError("PEP8 Format not respected")


def print_files_information_pep8():
    """
    Print the list of files which can be removed from the whitelist and the
    list of files which do not respect PEP8 formatting that aren't in the
    whitelist
    """
    infracting_files = []
    non_infracting_files = []
    pep8_checker = StyleGuide(quiet=True)
    for path in list_files(".py"):
        number_of_infractions = pep8_checker.input_file(path)
        rel_path = os.path.relpath(path, pylearn2.__path__[0])
        if number_of_infractions > 0:
            if rel_path not in whitelist_pep8:
                infracting_files.append(path)
        else:
            if rel_path in whitelist_pep8:
                non_infracting_files.append(path)
    print("Files that must be corrected or added to whitelist:")
    for file in infracting_files:
        print(file)
    print("Files that can be removed from whitelist:")
    for file in non_infracting_files:
        print(file)


def test_format_docstrings():
    """
    Test if docstrings are well formatted.
    """

    try:
        verify_format_docstrings()
    except SkipTest as e:
        import traceback
        traceback.print_exc(e)
        raise AssertionError(
            "Some file raised SkipTest on import, and inadvertently"
            " canceled the documentation testing."
        )


def verify_format_docstrings():
    """
    Implementation of `test_format_docstrings`. The implementation is
    factored out so it can be placed inside a guard against SkipTest.
    """
    format_infractions = []

    for path in list_files(".py"):
        rel_path = os.path.relpath(path, pylearn2.__path__[0])
        if rel_path in whitelist_docstrings:
            continue
        try:
            format_infractions.extend(docstring_errors(path))
        except Exception as e:
            format_infractions.append(["%s failed to run so format cannot "
                                       "be checked. Error message:\n %s" %
                                       (rel_path, e)])

    if len(format_infractions) > 0:
        msg = "\n".join(':'.join(line) for line in format_infractions)
        raise AssertionError("Docstring format not respected:\n%s" % msg)

if __name__ == "__main__":
    print_files_information_pep8()
