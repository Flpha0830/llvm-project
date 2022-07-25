#include "RegAllocGreedy.h"
#include "AllocationOrder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/Analysis/TensorSpec.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ReleaseModeModelRunner.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"

#include "llvm/Analysis/ModelUnderTrainingRunner.h"
#include "llvm/Analysis/NoInferenceModelRunner.h"

#include "RegAllocScore.h"
#include "llvm/Analysis/Utils/TFUtils.h"


using namespace llvm;

static cl::opt<std::string> TrainingLog(
    "regalloc-prio-training-log", cl::Hidden,
    cl::desc("Training log for the register allocator priority model"));

static cl::opt<std::string> ModelUnderTraining(
     "regalloc-prio-model", cl::Hidden,
     cl::desc("The model being trained for register allocation priority"));

namespace llvm {

static const std::vector<int64_t> PerLiveRangeShape{1};

#define RA_PRIORITY_FEATURES_LIST(M)                                              \
  M(int64_t, size, PerLiveRangeShape, "size")                                         \
  M(int64_t, stage, PerLiveRangeShape, "stage")                        \
  M(float, weight, PerLiveRangeShape, "weight")                               \

#define DecisionName "priority"

// Named features index.
enum FeatureIDs {
#define _FEATURE_IDX(_, name, __, ___) name,
  RA_PRIORITY_FEATURES_LIST(_FEATURE_IDX)
#undef _FEATURE_IDX
      FeatureCount
};

class MLPriorityAdvisor : public RegAllocPriorityAdvisor {
public:
  MLPriorityAdvisor(const MachineFunction &MF, const RAGreedy &RA,
                 MLModelRunner *Runner);
  void logReward(float reward) override {}

protected:
  const RegAllocPriorityAdvisor &getDefaultAdvisor() const {
    return static_cast<const RegAllocPriorityAdvisor &>(DefaultAdvisor);
  }

  // The assumption is that if the Runner could not be constructed, we emit-ed
  // error, and we shouldn't be asking for it here.
  const MLModelRunner &getRunner() const { return *Runner; }

  float tryFindPriority(unsigned Prio, unsigned Size, LiveRangeStage Stage, float Weight) const override;

private:
  const DefaultPriorityAdvisor DefaultAdvisor;
  MLModelRunner *const Runner;
};

#define _DECL_FEATURES(type, name, shape, _)                                   \
  TensorSpec::createSpec<type>(#name, shape),

static const std::vector<TensorSpec> InputFeatures{
    {RA_PRIORITY_FEATURES_LIST(_DECL_FEATURES)},
};
#undef _DECL_FEATURES

// ===================================
// Release (AOT) - specifics
// ===================================
class ReleaseModePriorityAdvisorAnalysis final
    : public RegAllocPriorityAdvisorAnalysis {
public:
  ReleaseModePriorityAdvisorAnalysis()
      : RegAllocPriorityAdvisorAnalysis(AdvisorMode::Release) {}
  // support for isa<> and dyn_cast.
  static bool classof(const RegAllocPriorityAdvisorAnalysis *R) {
    return R->getAdvisorMode() == AdvisorMode::Release;
  }

private:
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    RegAllocPriorityAdvisorAnalysis::getAnalysisUsage(AU);
  }

  std::unique_ptr<RegAllocPriorityAdvisor>
  getAdvisor(const MachineFunction &MF, const RAGreedy &RA) override {
    if (!Runner)
      Runner = std::make_unique<ReleaseModeModelRunner<NoopSavedModelImpl>>(
          MF.getFunction().getContext(), InputFeatures, DecisionName);
    return std::make_unique<MLPriorityAdvisor>(
        MF, RA, Runner.get());
  }
  std::unique_ptr<ReleaseModeModelRunner<NoopSavedModelImpl>> Runner;
};


static const TensorSpec Output =
    TensorSpec::createSpec<float>(DecisionName, {1});
static const TensorSpec Reward = TensorSpec::createSpec<float>("reward", {1});


#define _DECL_TRAIN_FEATURES(type, name, shape, _)                             \
  TensorSpec::createSpec<type>(std::string("action_") + #name, shape),

static const std::vector<TensorSpec> TrainingInputFeatures{
    {RA_PRIORITY_FEATURES_LIST(_DECL_TRAIN_FEATURES)
         TensorSpec::createSpec<float>("action_discount", {1}),
     TensorSpec::createSpec<int32_t>("action_step_type", {1}),
     TensorSpec::createSpec<float>("action_reward", {1})}};
#undef _DECL_TRAIN_FEATURES

class DevelopmentModePriorityAdvisor : public MLPriorityAdvisor {
public:
    DevelopmentModePriorityAdvisor(const MachineFunction &MF, const RAGreedy &RA,
                                MLModelRunner *Runner,
                                Logger *Log)
        : MLPriorityAdvisor(MF, RA, Runner), Log(Log) {}

    void logReward(float reward) override {
        Log->logFloatFinalReward(reward);
    }

private:
  float tryFindPriority(unsigned Prio, unsigned Size, LiveRangeStage Stage, float Weight) const override;
  Logger *const Log;
};

class DevelopmentModePriorityAdvisorAnalysis final
    : public RegAllocPriorityAdvisorAnalysis {
public:
  DevelopmentModePriorityAdvisorAnalysis()
      : RegAllocPriorityAdvisorAnalysis(AdvisorMode::Development) {}
  // support for isa<> and dyn_cast.
  static bool classof(const RegAllocPriorityAdvisorAnalysis *R) {
    return R->getAdvisorMode() == AdvisorMode::Development;
  }

  /// get the logger for the given function, or nullptr if we didn't collect
  /// one. This is used to inject the score by the RegAllocScoring pass.
  Logger *getLogger(const MachineFunction &MF) const {
    auto I = LogMap.find(MF.getName());
    if (I == LogMap.end())
      return nullptr;
    return I->second.get();
  }

private:
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    RegAllocPriorityAdvisorAnalysis::getAnalysisUsage(AU);
  }

  // Save all the logs (when requested).
  bool doFinalization(Module &M) override {
    if (TrainingLog.empty())
      return false;
    std::error_code EC;
    auto OS = std::make_unique<raw_fd_ostream>(TrainingLog, EC);
    if (EC) {
      M.getContext().emitError(EC.message() + ":" + TrainingLog);
      return false;
    }
    Logger::flushLogs(*OS, LogMap);
    return false;
  }

  std::unique_ptr<RegAllocPriorityAdvisor>
  getAdvisor(const MachineFunction &MF, const RAGreedy &RA) override {

    LLVMContext &Ctx = MF.getFunction().getContext();
    if (ModelUnderTraining.empty() && TrainingLog.empty()) {
      Ctx.emitError("Regalloc development mode should be requested with at "
                    "least logging enabled and/or a training model");
      return nullptr;
    }
    if (!Runner) {
      if (ModelUnderTraining.empty())
        Runner = std::make_unique<NoInferenceModelRunner>(Ctx, InputFeatures);
      else
        Runner = ModelUnderTrainingRunner::createAndEnsureValid(
            Ctx, ModelUnderTraining, DecisionName, TrainingInputFeatures);
      if (!Runner) {
        Ctx.emitError("Regalloc: could not set up the model runner");
        return nullptr;
      }
    }

    Logger *Log = nullptr;
    if (!TrainingLog.empty()) {
      std::vector<LoggedFeatureSpec> LFS;
      for (const auto &FS : InputFeatures)
        LFS.push_back({FS, None});
      if (auto *MUTR = dyn_cast<ModelUnderTrainingRunner>(Runner.get()))
        if (MUTR->outputLoggedFeatureSpecs().size() > 1)
          append_range(LFS, drop_begin(MUTR->outputLoggedFeatureSpecs()));
      // We always log the output; in particular, if we're not evaluating, we
      // don't have an output spec json file. That's why we handle the
      // 'normal' output separately.
      LFS.push_back({Output, None});
      auto I = LogMap.insert(std::make_pair(
          MF.getFunction().getName(),
          std::make_unique<Logger>(LFS, Reward, /*IncludeReward*/ true)));
      assert(I.second);
      Log = I.first->second.get();
    }
    
    return std::make_unique<DevelopmentModePriorityAdvisor>(
        MF, RA, Runner.get(), Log);
  }

  std::unique_ptr<MLModelRunner> Runner;
  StringMap<std::unique_ptr<Logger>> LogMap;
};

}

RegAllocPriorityAdvisorAnalysis *llvm::createDevelopmentModePriorityAdvisor() {
  return new DevelopmentModePriorityAdvisorAnalysis();
}

RegAllocPriorityAdvisorAnalysis *llvm::createReleaseModePriorityAdvisor() {
  return new ReleaseModePriorityAdvisorAnalysis();
}


MLPriorityAdvisor::MLPriorityAdvisor(const MachineFunction &MF, const RAGreedy &RA,
                               MLModelRunner *Runner)
    : RegAllocPriorityAdvisor(MF, RA), DefaultAdvisor(MF, RA),
      Runner(std::move(Runner)) {
  assert(this->Runner);
}


float MLPriorityAdvisor::tryFindPriority(unsigned Prio, unsigned Size, LiveRangeStage Stage, float Weight) const {
    *Runner->getTensor<int64_t>(0) = static_cast<int64_t>(Size);
    *Runner->getTensor<int64_t>(1) = static_cast<int64_t>(Stage);
    *Runner->getTensor<float>(2) = static_cast<float>(Weight);

    float Ret = Runner->evaluate<float>();
    return Ret;
}

float DevelopmentModePriorityAdvisor::tryFindPriority(unsigned Prio, unsigned Size, LiveRangeStage Stage, float Weight) const {
    float Ret = 0;

    if (isa<ModelUnderTrainingRunner>(getRunner())) {
        Ret = MLPriorityAdvisor::tryFindPriority(Prio, Size, Stage, Weight);
    } else {
        Ret = static_cast<float>(Prio);
    }

    if (TrainingLog.empty())
        return Ret;

    size_t CurrentFeature = 0;
    for (; CurrentFeature < InputFeatures.size(); ++CurrentFeature) {
        Log->logSpecifiedTensorValue(
            CurrentFeature, reinterpret_cast<const char *>(
                                getRunner().getTensorUntyped(CurrentFeature)));
    }

    if (auto *MUTR = dyn_cast<ModelUnderTrainingRunner>(&getRunner())) {
      for (size_t I = 1; I < MUTR->outputLoggedFeatureSpecs().size();
          ++I, ++CurrentFeature)
        Log->logSpecifiedTensorValue(
            CurrentFeature,
            reinterpret_cast<const char *>(
                MUTR->lastEvaluationResult()->getUntypedTensorValue(I)));
    }
    Log->logFloatValue(CurrentFeature, &Ret);

    return Ret;
}