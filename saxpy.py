#!usr/bin/env /home/clu/py38/bin/python

import numpy as np
import pyopencl as cl

# a_np = np.random.rand(50000).astype(np.float32)
# b_np = np.random.rand(50000).astype(np.float32)

a = np.float32(10)
X_np = np.arange(0,10,1).astype(np.float32)
Y_np = np.arange(0,10,1).astype(np.float32)

print(a)
print(X_np)
print(Y_np)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
# a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
# b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

X_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X_np)
Y_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=Y_np)

prg = cl.Program(ctx, """
__kernel void saxpy(const __global float * X,
                        __global float * Y,
                  const float a)
{
  uint gid = get_global_id(0);
  Y[gid] = a* X[gid] + Y[gid];
}

""").build()

# res_g = cl.Buffer(ctx, mf.WRITE_ONLY, Y_np.nbytes)
knl = prg.saxpy  # Use this Kernel object for repeated calls

# print(knl)
# knl.set_args(X_g, Y_g, a)
# cl.enqueue_nd_range_kernel(queue, knl, Y_np.shape, None)

knl(queue, Y_np.shape, None, X_g, Y_g, a)

res_np = np.empty_like(Y_np)
print(res_np)
cl.enqueue_copy(queue, res_np, Y_g)
print(res_np)

# Check on CPU with Numpy:
# print(res_np - (a + b_np))
# print(np.linalg.norm(res_np - (a + b_np)))
# assert np.allclose(res_np, a + b_np)
