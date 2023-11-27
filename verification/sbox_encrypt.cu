extern "C" {

#include <stdint.h>

#define SBOXSIZE (256)
__global__ void kernel_sbox_encrypt(const uint32_t *plainTextBlocks, uint32_t *cipherTextBlocks, const uint32_t *key, const uint32_t *_sbox, uint32_t *runtime)
{
    __shared__ uint32_t sbox[SBOXSIZE];
    
    int tid = blockIdx.x *blockDim.x + threadIdx.x;
    int stride = 32;
    
    // init shared memory since CUDA disallows static initialization of shared memory
    // using thread to copy SBOX passed in before starting timing
    sbox[tid+stride*0] = _sbox[tid+stride*0]; // 0-31
    sbox[tid+stride*1] = _sbox[tid+stride*1]; // 32-63
    sbox[tid+stride*2] = _sbox[tid+stride*2]; // 64-96 ... etc
    sbox[tid+stride*3] = _sbox[tid+stride*3];
    sbox[tid+stride*4] = _sbox[tid+stride*4];
    sbox[tid+stride*5] = _sbox[tid+stride*5];
    sbox[tid+stride*6] = _sbox[tid+stride*6];
    sbox[tid+stride*7] = _sbox[tid+stride*7]; // 224-255

    register uint32_t tmp = clock();
    
    // Perform sbox encryption for each plain text block
    // Takes each byte and performs lookup to SBOX followed by XOR with key
    cipherTextBlocks[tid] = 
    (sbox[(plainTextBlocks[tid] & 0x000000FF)]) ^
    (sbox[(plainTextBlocks[tid] & 0x0000FF00) >> 8]) ^
    (sbox[(plainTextBlocks[tid] & 0x00FF0000) >> 16]) ^
    (sbox[(plainTextBlocks[tid] & 0xFF000000) >> 24]) ^
    key[0];

    __syncwarp();
    runtime[0] = clock() - tmp;
}

}
