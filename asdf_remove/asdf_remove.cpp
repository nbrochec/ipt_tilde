
#include "c74_min.h"

using namespace c74::min;


class playground : public c74::min::object<playground> {
private:
    // TODO: Member variables here 

public:
    MIN_DESCRIPTION{"Playground class"};
    MIN_TAGS{""};
    MIN_AUTHOR{""};
    MIN_RELATED{""};

    inlet<> inlet_main{this, "(any) input"};

    outlet<> outlet_main{this, "(any) output"};
    outlet<> dumpout{this, "(any) dumpout"};
    
};

MIN_EXTERNAL(playground)