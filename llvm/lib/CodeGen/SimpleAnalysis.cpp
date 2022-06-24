

#include "SimpleAnalysis.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineFunction.h"

using namespace llvm;


char SimpleAnalysis::ID = 0;
INITIALIZE_PASS(SimpleAnalysis, "regalloc-simple",
                "simple analysis", true, true)

StringRef SimpleAnalysis::getPassName() const {
    return "Simple Analysis";
}

void SimpleAnalysis::printmf(const MachineFunction &MF) {
    MF.dump();
    printf("Hello, I'm a Simple Analysis!!");
}
