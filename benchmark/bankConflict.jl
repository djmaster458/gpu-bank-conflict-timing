using Test

using CUDA
using Plots
using Statistics

println(CUDA.versioninfo())
println("Blocks: 1")
println("Warpsize: ", CUDA.warpsize(CUDA.device()))

md = CuModuleFile(joinpath(@__DIR__, "bank_conflict.ptx"))
bank_conflict = CuFunction(md, "kernel_bank_conflict")

nthreads = CUDA.warpsize(CUDA.device())
nblocks = 1

average_times_per_stride = zeros(Float32, 32)

for i in 1:10_000
    for j in 1:32
        stride = Float32[j]
        times = zeros(Float32, nthreads)
        out = zeros(Float32, nthreads)
        
        d_stride = CuArray(stride)
        d_times = CuArray(times)
        d_out = CuArray(out)
        
        CUDA.@sync begin
            cudacall(bank_conflict, Tuple{CuPtr{Cint},CuPtr{Cfloat},CuPtr{Cfloat}}, d_stride, d_out, d_times; threads=nthreads, blocks=nblocks)
        end

        out_times = Array(d_times)
        average_times_per_stride[j] += Statistics.mean(out_times)
    end
end

println("GPU Done")

average_times_per_stride = average_times_per_stride ./ 10_000

println(average_times_per_stride)

histogram(average_times_per_stride, legend=false, xlabel="GPU Clock Cycles", ylabel="Frequency")
