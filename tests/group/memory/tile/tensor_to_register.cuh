#include "testing_flags.cuh"

#if defined(TEST_GROUP_MEMORY_TILE_TENSOR_TO_REGISTER) && defined(KITTENS_BLACKWELL)

#include "testing_commons.cuh"

namespace group {
namespace memory {
namespace tile {
namespace tensor_to_register {

void tests(test_data &results);

}
}
}
}

#endif
