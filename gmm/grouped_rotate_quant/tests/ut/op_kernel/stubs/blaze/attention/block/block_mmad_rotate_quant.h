#pragma once
namespace Blaze::Attention {
template<class = void> struct BlockRotateQuant {};
namespace Block {
template<class Policy, class A, class LA, class B, class LB, class Out, class LC, class C, class LO>
struct BlockMmad {};
}
}
