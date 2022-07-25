#include "llvm/Pass.h"

namespace llvm {

class MachineFunction;
class RAGreedy;

class RegAllocPriorityAdvisor {
public:
  RegAllocPriorityAdvisor(const RegAllocPriorityAdvisor &) = delete;
  RegAllocPriorityAdvisor(RegAllocPriorityAdvisor &&) = delete;
  virtual ~RegAllocPriorityAdvisor() = default;

  virtual float tryFindPriority(unsigned Prio, unsigned Size, LiveRangeStage Stage, float Weight) const = 0;
  virtual void logReward(float reward) = 0;

protected:
  RegAllocPriorityAdvisor(const MachineFunction &MF, const RAGreedy &RA);

  const MachineFunction &MF;
  const RAGreedy &RA;
};


class RegAllocPriorityAdvisorAnalysis : public ImmutablePass {
public:
  enum class AdvisorMode : int { Default, Release, Development };

  RegAllocPriorityAdvisorAnalysis(AdvisorMode Mode)
      : ImmutablePass(ID), Mode(Mode){};
  static char ID;

  /// Get an advisor for the given context (i.e. machine function, etc)
  virtual std::unique_ptr<RegAllocPriorityAdvisor>
  getAdvisor(const MachineFunction &MF, const RAGreedy &RA) = 0;
  AdvisorMode getAdvisorMode() const { return Mode; }

protected:
  // This analysis preserves everything, and subclasses may have additional
  // requirements.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

private:
  StringRef getPassName() const override;
  const AdvisorMode Mode;
};

template <> Pass *callDefaultCtor<RegAllocPriorityAdvisorAnalysis>();

RegAllocPriorityAdvisorAnalysis *createReleaseModePriorityAdvisor();

RegAllocPriorityAdvisorAnalysis *createDevelopmentModePriorityAdvisor();

class DefaultPriorityAdvisor : public RegAllocPriorityAdvisor {
public:
  DefaultPriorityAdvisor(const MachineFunction &MF, const RAGreedy &RA)
      : RegAllocPriorityAdvisor(MF, RA) {}

  void logReward(float reward) override {};

private:
  float tryFindPriority(unsigned Prio, unsigned Size, LiveRangeStage Stage, float Weight) const override;
};
}