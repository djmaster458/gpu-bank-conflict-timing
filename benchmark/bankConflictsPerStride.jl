function MemoryIndexToBankIndex(stride::Int)
# Since banksize is 4 bytes, each 4-byte array element belongs to consecutive banks
# This is until the wrap around occurs at index 32
# So arr[0] => bank 0, arr[1] => bank 1, ...
# arr[31] => bank 31, but arr[32] => bank 0
# to calculate bank conflicts, we can simply take the memory index mod 32
# we can find all of bank access for a given stride this way for a warp of 32 threads
    nthreads = 32
    number_of_banks = 32
    bank_indices = zeros(Int, nthreads)

    for tid in 0:nthreads-1
        # Juila Arrays start at 1
        bank_indices[tid+1] = (stride * tid) % number_of_banks
    end

    unique_elements = unique(bank_indices)
    number_of_conflicts = length(bank_indices) - length(unique_elements)

    return bank_indices, number_of_conflicts
end

stride_by_conflicts = zeros(Int, 32)

for stride in 1:32
    _, number_of_conflicts = MemoryIndexToBankIndex(stride)
    stride_by_conflicts[stride] = number_of_conflicts
end

strides = range(1,32)
scatter(strides, stride_by_conflicts, xlabel="Stride", ylabel="Number of Conflicts")