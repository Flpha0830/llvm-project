#include "llvm/Pass.h"

namespace llvm {

class MachineFunction;
class RAGreedy;

class SimpleAnalysis : public ImmutablePass {
public:
  SimpleAnalysis()
      : ImmutablePass(ID){};
  static char ID;

  void printmf(const MachineFunction &MF);

protected:
  // This analysis preserves everything, and subclasses may have additional
  // requirements.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

private:
StringRef getPassName() const override;
};
}