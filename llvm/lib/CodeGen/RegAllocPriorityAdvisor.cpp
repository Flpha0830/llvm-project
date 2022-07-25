#include "RegAllocGreedy.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineFunction.h"

using namespace llvm;

static cl::opt<RegAllocPriorityAdvisorAnalysis::AdvisorMode> Mode(
    "regalloc-prio-enable-advisor", cl::Hidden,
    cl::init(RegAllocPriorityAdvisorAnalysis::AdvisorMode::Default),
    cl::desc("Enable regalloc advisor mode"),
    cl::values(
        clEnumValN(RegAllocPriorityAdvisorAnalysis::AdvisorMode::Default,
                   "default", "Default"),
        clEnumValN(RegAllocPriorityAdvisorAnalysis::AdvisorMode::Release,
                   "release", "precompiled"),
        clEnumValN(RegAllocPriorityAdvisorAnalysis::AdvisorMode::Development,
                   "development", "for training")));

char RegAllocPriorityAdvisorAnalysis::ID = 0;
INITIALIZE_PASS(RegAllocPriorityAdvisorAnalysis, "regalloc-prio",
                "RegAllocPriorityAdvisorAnalysis", false, true)

namespace {
class DefaultPriorityAdvisorAnalysis final
    : public RegAllocPriorityAdvisorAnalysis {
public:
  DefaultPriorityAdvisorAnalysis(bool NotAsRequested)
      : RegAllocPriorityAdvisorAnalysis(AdvisorMode::Default),
        NotAsRequested(NotAsRequested) {}

  // support for isa<> and dyn_cast.
  static bool classof(const RegAllocPriorityAdvisorAnalysis *R) {
    return R->getAdvisorMode() == AdvisorMode::Default;
  }

private:
  std::unique_ptr<RegAllocPriorityAdvisor>
  getAdvisor(const MachineFunction &MF, const RAGreedy &RA) override {
    return std::make_unique<DefaultPriorityAdvisor>(MF, RA);
  }
  bool doInitialization(Module &M) override {
    if (NotAsRequested)
      M.getContext().emitError("Requested regalloc priority advisor analysis "
                               "could be created. Using default");
    return RegAllocPriorityAdvisorAnalysis::doInitialization(M);
  }
  const bool NotAsRequested;
};
} // namespace

template <> Pass *llvm::callDefaultCtor<RegAllocPriorityAdvisorAnalysis>() {
  Pass *Ret = nullptr;
  switch (Mode) {
  case RegAllocPriorityAdvisorAnalysis::AdvisorMode::Default:
    Ret = new DefaultPriorityAdvisorAnalysis(/*NotAsRequested*/ false);
    break;
  case RegAllocPriorityAdvisorAnalysis::AdvisorMode::Development:
#if defined(LLVM_HAVE_TF_API)
    Ret = createDevelopmentModePriorityAdvisor();
#endif
    break;
  case RegAllocPriorityAdvisorAnalysis::AdvisorMode::Release:
#if defined(LLVM_HAVE_TF_AOT)
    Ret = createReleaseModePriorityAdvisor();
#endif
    break;
  }
  if (Ret)
    return Ret;
  return new DefaultPriorityAdvisorAnalysis(/*NotAsRequested*/ true);
}

StringRef RegAllocPriorityAdvisorAnalysis::getPassName() const {
  switch (getAdvisorMode()) {
  case AdvisorMode::Default:
    return "Default Regalloc Priority Advisor";
  case AdvisorMode::Release:
    return "Release mode Regalloc Priority Advisor";
  case AdvisorMode::Development:
    return "Development mode Regalloc Priority Advisor";
  }
  llvm_unreachable("Unknown advisor kind");
}

RegAllocPriorityAdvisor::RegAllocPriorityAdvisor(const MachineFunction &MF,
                                                 const RAGreedy &RA): MF(MF), RA(RA) {}

float DefaultPriorityAdvisor::tryFindPriority(unsigned Prio, unsigned Size, LiveRangeStage Stage, float Weight) const {
    return static_cast<float>(Prio);
}