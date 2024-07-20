from dataclasses import fields
from typing import Any
from typing import Callable
from typing import List

import jax.nn.initializers as inits
import jax.numpy as jnp
from flax import struct
from jax import Array

PRNGKey = Any
Shape = List[int]
Dtype = Any
Initializer = Callable[[PRNGKey, Shape, Dtype], Array]


# dataclasses 提供了一个 @dataclass 装饰器，通过它，可以极大地简化类的定义.不需要编写繁琐的 __init__、__repr__ 等方法
# dataclasses模块定义了几个为我们提供了一种更简洁、更优雅地定义数据类的方式。后面的则是在dataclasses语法下定义的类型
@struct.dataclass
class TransformerConfig:
    param_dtype: Dtype
    dtype: Dtype
    global_batch_size: int
    sequence_len: int
    update_len: int
    block_len: int
    mem_len: int
    grad_thru_cache: bool
    agg_cache: bool
    d_model: int
    d_k: int
    d_v: int
    d_ff: int
    n_head: int
    n_code: int
    n_layer: int
    n_vocab: int
    pe_abs: bool
    pe_lam: float
    p_dropemb: float
    p_dropsin: float
    p_dropres: float
    p_droplyr: float
    p_nucleus: float
    c_beta: float
    c_gamma: float
    e_tie: bool
    e_preln: bool
    e_scale: str
    is_train: bool
    e_init: Initializer
    w_init: Initializer
    r_init: Initializer
    b_init: Initializer
    no_emb: bool = False

    @classmethod
    def create(cls, **kwargs):
        signature = {field.name: field.type for field in fields(TransformerConfig)} # 这里的field是一个包
        filtered = {k: v for k, v in kwargs.items() if k in signature}
# 从字典流中过滤掉不属于这个类的标签值，只留下和这个类相关的键值对
        if isinstance(filtered["param_dtype"], str): # 针对第一个成员，更改外界输入的类别，从字符串变成Dtype
            filtered["param_dtype"] = jnp.dtype(filtered["param_dtype"])

        if isinstance(filtered["dtype"], str): #同上
            filtered["dtype"] = jnp.dtype(filtered["dtype"])

        for k, v in filtered.items():
            if signature[k] is bool and v in {0, 1}:
                filtered[k] = bool(v) # 把0和1实数值转化成布尔值

        filtered["e_init"] = inits.normal(1.0) #这四个数都用jax的函数初始化，不会的函数去查JAX库
        filtered["w_init"] = inits.variance_scaling(1.0, "fan_in", "normal")
        filtered["r_init"] = inits.variance_scaling(1.0, "fan_in", "normal")
        filtered["b_init"] = inits.zeros

        return cls(**filtered)
'''
在Python中，@classmethod 是一个装饰器，用于将一个方法定义为类方法。类方法是一种特殊类型的方法，它属于类而不是类的实例。
这意味着类方法可以被类本身调用，也可以被类的实例调用，但它们的第一个参数总是指向类本身，而不是类的实例。
函数里面cls指的是类本身
'''