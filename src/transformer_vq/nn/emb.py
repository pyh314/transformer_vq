import dataclasses

import flax.linen as nn
import jax.numpy as jnp

from transformer_vq.nn.types import TransformerConfig


class Embeddings(nn.Module): # 所有的层都是nn.Module类的子层
    config: TransformerConfig # 这一块是直接将TransformetConfig类当成Embeddings类的数据成员了

    def setup(self):
        self.apply_config()
        emb_args = [self.e_init, [self.n_vocab, self.d_model], self.param_dtype] #列表内部的元素可以不是同一种数字类型，所以列表里面套列表也不稀奇
        self.embs = self.param("embs", *emb_args)
        bias_out_args = [self.b_init, [self.n_vocab], self.param_dtype]
        self.bias_out = self.param("bias_out", *bias_out_args)

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    def __call__(self, x):
        x = jnp.take_along_axis(
            self.embs[None, ...], x[..., None].astype(jnp.int32), axis=1 
        ) 
        return x.astype(self.dtype)

    def logits(self, x):
        x = x.astype(jnp.float32)
        x = jnp.dot(x, self.embs.T.astype(jnp.float32))
        x += self.bias_out.astype(jnp.float32)[None, None, ...]
        return x
'''
[..., None]：这种语法是 NumPy 的一个特性，称为“新轴”（new axis）。它在数组的形状中添加一个新的轴。
例如，如果 pos_seq 的形状是 (10,)（一个有10个元素的一维数组），使用 [..., None] 后，它的形状变为 (1, 10)，
即在第一个维度前添加了一个大小为1的新维度。
'''