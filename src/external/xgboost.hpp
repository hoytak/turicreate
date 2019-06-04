#ifndef TURI_EXTERNAL_XGBOOST_HPP_
#define TURI_EXTERNAL_XGBOOST_HPP_

#include <external/xgboost/R-package/src/xgboost_R.h>
#include <external/xgboost/wrapper/xgboost_wrapper.h>
#include <external/xgboost/java/xgboost4j_wrapper.h>
#include <external/xgboost/subtree/rabit/wrapper/rabit_wrapper.h>
#include <external/xgboost/subtree/rabit/include/rabit_serializable.h>
#include <external/xgboost/subtree/rabit/include/rabit.h>
#include <external/xgboost/subtree/rabit/include/rabit/utils.h>
#include <external/xgboost/subtree/rabit/include/rabit/engine.h>
#include <external/xgboost/subtree/rabit/include/rabit/rabit-inl.h>
#include <external/xgboost/subtree/rabit/include/rabit/timer.h>
#include <external/xgboost/subtree/rabit/include/rabit/io.h>
#include <external/xgboost/subtree/rabit/include/dmlc/io.h>
#include <external/xgboost/subtree/rabit/src/allreduce_robust.h>
#include <external/xgboost/subtree/rabit/src/allreduce_mock.h>
#include <external/xgboost/subtree/rabit/src/allreduce_robust-inl.h>
#include <external/xgboost/subtree/rabit/src/socket.h>
#include <external/xgboost/subtree/rabit/src/allreduce_base.h>
#include <external/xgboost/src/data.h>
#include <external/xgboost/src/tree/updater_skmaker-inl.hpp>
#include <external/xgboost/src/tree/updater_basemaker-inl.hpp>
#include <external/xgboost/src/tree/updater_refresh-inl.hpp>
#include <external/xgboost/src/tree/updater_histmaker-inl.hpp>
#include <external/xgboost/src/tree/updater_colmaker-inl.hpp>
#include <external/xgboost/src/tree/updater.h>
#include <external/xgboost/src/tree/param.h>
#include <external/xgboost/src/tree/model.h>
#include <external/xgboost/src/tree/updater_prune-inl.hpp>
#include <external/xgboost/src/tree/updater_distcol-inl.hpp>
#include <external/xgboost/src/tree/updater_sync-inl.hpp>
#include <external/xgboost/src/io/libsvm_parser.h>
#include <external/xgboost/src/io/page_dmatrix-inl.hpp>
#include <external/xgboost/src/io/simple_dmatrix-inl.hpp>
#include <external/xgboost/src/io/io.h>
#include <external/xgboost/src/io/simple_fmatrix-inl.hpp>
#include <external/xgboost/src/io/sparse_batch_page.h>
#include <external/xgboost/src/io/page_fmatrix-inl.hpp>
#include <external/xgboost/src/utils/utils.h>
#include <external/xgboost/src/utils/config.h>
#include <external/xgboost/src/utils/group_data.h>
#include <external/xgboost/src/utils/quantile.h>
#include <external/xgboost/src/utils/thread.h>
#include <external/xgboost/src/utils/io.h>
#include <external/xgboost/src/utils/fmap.h>
#include <external/xgboost/src/utils/omp.h>
#include <external/xgboost/src/utils/base64-inl.h>
#include <external/xgboost/src/utils/math.h>
#include <external/xgboost/src/utils/bitmap.h>
#include <external/xgboost/src/utils/thread_buffer.h>
#include <external/xgboost/src/utils/iterator.h>
#include <external/xgboost/src/utils/random.h>
#include <external/xgboost/src/learner/helper_utils.h>
#include <external/xgboost/src/learner/objective-inl.hpp>
#include <external/xgboost/src/learner/evaluation.h>
#include <external/xgboost/src/learner/objective.h>
#include <external/xgboost/src/learner/dmatrix.h>
#include <external/xgboost/src/learner/evaluation-inl.hpp>
#include <external/xgboost/src/learner/learner-inl.hpp>
#include <external/xgboost/src/gbm/gblinear-inl.hpp>
#include <external/xgboost/src/gbm/gbtree-inl.hpp>
#include <external/xgboost/src/gbm/gbm.h>
#include <external/xgboost/src/sync/sync.h>

#endif
