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
# 在Python中，dataclasses.asdict 是一个函数，它的作用是将一个带有 dataclass 装饰器的类的实例转换成一个字典。
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
在Python中，setattr 是一个内置函数，用于设置对象的属性值。你可以使用 setattr 为任何对象的属性赋值，即使这些属性在对象的类中没有明确定义
setattr(object, name, value):object: 要设置属性的对象。name: 要设置的属性的名称，通常以字符串形式提供。value: 要分配给属性的值。\
使用场景：当你需要动态地为对象添加属性或更改现有属性的值时。当你不确定属性是否存在，但需要为其赋值时。
'''