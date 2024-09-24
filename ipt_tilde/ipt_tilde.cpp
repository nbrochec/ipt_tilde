#include "c74_min.h"

using namespace c74::min;


class ipt_tilde : public c74::min::object<ipt_tilde> {
private:
    // TODO: Member variables here

public:
    MIN_DESCRIPTION{""};                   // TODO
    MIN_TAGS{""};                          // TODO
    MIN_AUTHOR{""};                        // TODO
    MIN_RELATED{""};                       // TODO

    inlet<> inlet_main{this, "(int) left operand", ""}; // TODO

    outlet<> outlet_main{this, "output", ""}; // TODO
    outlet<> dumpout{this, "(any) dumpout"};
};


MIN_EXTERNAL(ipt_tilde);