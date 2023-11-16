using CUDA
using Plots
using Statistics
using PrettyTables

println(CUDA.versioninfo())
println("Blocks: 1")
println("Warpsize: ", CUDA.warpsize(CUDA.device()))

md = CuModuleFile(joinpath(@__DIR__, "bank_conflict.ptx"))
bank_conflict = CuFunction(md, "kernel_bank_conflict")

nthreads = CUDA.warpsize(CUDA.device())
nblocks = 1

average_times_per_stride = zeros(Float32, nthreads)

for i in 1:1_000
    for j in 1:32
        stride = Int32[j]
        times = zeros(UInt32, nthreads)
        out = zeros(Float32, nthreads)
        
        d_stride = CuArray(stride)
        d_times = CuArray(times)
        d_out = CuArray(out)
        
        CUDA.@sync begin
            cudacall(bank_conflict, Tuple{CuPtr{Cint},CuPtr{Cfloat},CuPtr{Cuint}}, d_stride, d_out, d_times; threads=nthreads, blocks=nblocks)
        end

        out_times = Array(d_times)
        average_times_per_stride[j] += Statistics.mean(out_times)
    end
end

println("GPU Done")

average_times_per_stride = average_times_per_stride ./ 1_000

println(average_times_per_stride)

## Generate Frequency Counts
# histogram(average_times_per_stride, legend=false, xlabel="GPU Clock Cycles", ylabel="Frequency")

## Generate Strides vs. GPU Clock Time
# strides = range(1, 32)
# scatter(strides, average_times_per_stride, xlabel="Stride Value", ylabel="GPU Clock Cycles")

## Generate Pretty Table with times and strides

strides = 1:1:32
table_data = hcat(strides, average_times_per_stride)
header = (["Stride", "GPU Clock Cycles"])
pretty_table(
    table_data;
    header = header,
    backend = Val(:latex))

