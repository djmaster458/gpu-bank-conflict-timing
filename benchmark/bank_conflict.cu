extern "C" {

#include <stdint.h>
__global__ void kernel_bank_conflict(const int *_stride, float *out, uint32_t *times)
{
    register uint32_t tmp, tmp2 = 0, offset = 32;
    __shared__ uint32_t share_data[1024*4];

    const int stride = _stride[0];
    
    int tid = blockIdx.x *blockDim.x + threadIdx.x;
    tmp = clock();

    tmp2 += share_data[tid*stride+0*offset];
    tmp2 += share_data[tid*stride+1*offset];
    tmp2 += share_data[tid*stride+2*offset];
    tmp2 += share_data[tid*stride+3*offset];
    tmp2 += share_data[tid*stride+4*offset];
    tmp2 += share_data[tid*stride+5*offset];
    tmp2 += share_data[tid*stride+6*offset];
    tmp2 += share_data[tid*stride+7*offset];
    tmp2 += share_data[tid*stride+8*offset];
    tmp2 += share_data[tid*stride+9*offset];
    tmp2 += share_data[tid*stride+10*offset];
    tmp2 += share_data[tid*stride+11*offset];
    tmp2 += share_data[tid*stride+12*offset];
    tmp2 += share_data[tid*stride+13*offset];
    tmp2 += share_data[tid*stride+14*offset];
    tmp2 += share_data[tid*stride+15*offset];
    tmp2 += share_data[tid*stride+16*offset];
    tmp2 += share_data[tid*stride+17*offset];
    tmp2 += share_data[tid*stride+18*offset];
    tmp2 += share_data[tid*stride+19*offset];
    tmp2 += share_data[tid*stride+20*offset];
    tmp2 += share_data[tid*stride+21*offset];
    tmp2 += share_data[tid*stride+22*offset];
    tmp2 += share_data[tid*stride+23*offset];
    tmp2 += share_data[tid*stride+24*offset];
    tmp2 += share_data[tid*stride+25*offset];
    tmp2 += share_data[tid*stride+26*offset];
    tmp2 += share_data[tid*stride+27*offset];
    tmp2 += share_data[tid*stride+28*offset];
    tmp2 += share_data[tid*stride+29*offset];
    tmp2 += share_data[tid*stride+30*offset];
    tmp2 += share_data[tid*stride+31*offset];
    tmp2 += share_data[tid*stride+32*offset];
    tmp2 += share_data[tid*stride+33*offset];
    tmp2 += share_data[tid*stride+34*offset];
    tmp2 += share_data[tid*stride+35*offset];
    tmp2 += share_data[tid*stride+36*offset];
    tmp2 += share_data[tid*stride+37*offset];
    tmp2 += share_data[tid*stride+38*offset];
    tmp2 += share_data[tid*stride+39*offset];
    
    times[tid] = clock() - tmp;
    out[tid] = tmp2;
}

}
