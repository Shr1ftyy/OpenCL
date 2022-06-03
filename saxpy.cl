kernel void saxpy(const __global float * X,
                        __global float * Y,
                  const float a)
{
  uint gid = get_global_id(0);
  Y[gid] = a* X[gid] + Y[gid];
}
