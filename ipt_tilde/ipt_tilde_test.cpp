// <algorithm> must precede the c74 headers: c74_min_limit.h does `using std::clamp`
// without including it, which MSVC's non-transitive STL rejects.
#include <algorithm>
#include "c74_min_unittest.h"
#include "ipt_tilde.cpp"


TEST_CASE("object is constructible") {
    ext_main(nullptr);

    test_wrapper<ipt_tilde> an_instance;
    ipt_tilde& obj = an_instance;

}
