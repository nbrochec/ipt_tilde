
#include "c74_min_unittest.h"
#include "asdf_remove.cpp"


TEST_CASE("object produces correct output") {
    ext_main(nullptr);

    test_wrapper<playground> an_instance;
    playground& playground = an_instance;

    REQUIRE(playground.inlets().size() == 1);
}


// ==============================================================================================
/**
 * Unit tests for shared code.
 * TODO Should ideally be moved to a separate file, but currently relies on a number of dependencies from c74::min
 *      and is therefore dependent on the c74::min cmake scripts as well as its modified catch2 framework at the moment
 */
// ==============================================================================================

TEST_CASE("") {
    // Unit tests for shared code go here
}