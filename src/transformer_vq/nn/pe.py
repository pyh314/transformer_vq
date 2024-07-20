import dataclasses
# 这里的pe指的是positional embeddings，位置编码
import chex
import flax.linen as nn
import jax.numpy as jnp

from transformer_vq.nn.types import TransformerConfig

# Chex is a library that provides decorators and functions to assert the shapes and types of JAX arrays and dataclasses.
def get_sinusoid_embs(length, width, lam, flip, start=0): #sinusoid embeddings 
    pos_seq = start + jnp.arange(length) # 位置编号0,1,2……
    chex.assert_shape(pos_seq, [length]) # Checks that the shape of all inputs matches specified expected_shapes.可以查阅chex文档 
    inv_lams = 1 / (lam ** (jnp.arange(0, width, 2) / width)) # 从0开始，每两个数生成一次，直到到width为止
    pre = pos_seq[..., None] * inv_lams[None, ...]
    sin = jnp.sin(pre)
    cos = jnp.cos(pre)
    cat = jnp.concatenate([sin, cos], axis=-1)
    chex.assert_shape(cat, [length, width])
    if not flip:
        return cat
    return jnp.flip(cat, axis=0)


class ScaledSin(nn.Module):
    # see w. hua et al., 2022
    config: TransformerConfig

    def setup(self):
        self.apply_config()
        self.scale = self.param("scale", self.b_init, [], jnp.float32)

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    def __call__(self, length, offset):
        embs = get_sinusoid_embs(
            length=length, start=offset, width=self.d_model, lam=self.pe_lam, flip=False
        )
        return (self.scale * embs).astype(self.dtype)
'''
[..., None]：这种语法是 NumPy 的一个特性，称为“新轴”（new axis）。它在数组的形状中添加一个新的轴。
例如，如果 pos_seq 的形状是 (10,)（一个有10个元素的一维数组），使用 [..., None] 后，它的形状变为 (1, 10)，即在第一个维度前添加了一个大小为1的新维度。
'''