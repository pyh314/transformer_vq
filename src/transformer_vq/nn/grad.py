import flax.linen as nn
import jax


def sg(x):
    return jax.lax.stop_gradient(x)


def st(x):
    return x - sg(x)


def maybe_remat(module, enabled):
    if enabled:
        return nn.remat(module)
    else:
        return module
'''
flax.linen.remat 是 Flax (JAX 的一个神经网络库) 中的一个功能，它允许你在神经网络中重新计算某些操作，而不是将它们的结果存储在内存中。
这种技术被称为“重新计算”（rematerialization），可以减少内存使用，特别是在处理大型模型或数据时非常有用。
减少内存使用：通过重新计算而不是存储中间结果，可以显著减少模型运行时的内存占用。
提高效率：在某些情况下，重新计算可能比从内存中加载中间结果更快，尤其是在内存带宽有限的情况下。
然而，重新计算可能会增加计算量，因为它需要多次执行某些操作。因此，需要权衡内存使用和计算时间。
重新计算的效果也取决于具体的模型和数据。在某些情况下，重新计算可能不会带来显著的性能提升。
'''