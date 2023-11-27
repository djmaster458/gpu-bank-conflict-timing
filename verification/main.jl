using CUDA;

const NBANKS = 32;
const BANKSIZE = 4;
const NLOOKUPS = 4;
const NTHREADS = 32;
const NBLOCKS = 1;
const NSAMPLES = 1_000_000;

function GenerateInverseSbox(sbox)
    # Create an map to store the inverse S-box
    is_box = Dict()

    # Fill map in the inverse S-box values
    for (i, val) in enumerate(sbox)
        is_box[val] = i
    end

    return is_box
end

function ConvertU32ToU8Array(val::UInt32)
    # Note this converts to machine endianness ordering
    return reinterpret(UInt8, [val]);
end

function CalculateNumberOfDuplicates(vec)
    # find number of duplicates (i.e. bank conflicts) give set of bank indices that access unique addresses
    return length(vec) - length(unique(vec));
end

function CalculateBankIndex(array_index)
    # Julia Array starts at 1, so we subtract 1 then modulo to get bank
    return (array_index - 1) % NBANKS;
end

function DetermineLookupTableAccesses(ciphertexts_by_thread, invSbox, key)
    # lookup instructions and their bytes used in the table lookup
    t1 = [];
    t2 = [];
    t3 = [];
    t4 = [];

    # based on section 4.3's equation
    for i in eachindex(ciphertexts_by_thread)
        block = ciphertexts_by_thread[i];
        block = block ⊻ key;
        lookup_block = invSbox[block];
        lookup_bytes = ConvertU32ToU8Array(lookup_block);

        # note: order matters based on endianness and order of lookups: t1 = first lookup = LSB
        push!(t1, lookup_bytes[1]);
        push!(t2, lookup_bytes[2]);
        push!(t3, lookup_bytes[3]);
        push!(t4, lookup_bytes[4]);
    end

    return t1, t2, t3, t4;
end

function FindBankConflictsPerLookupInst(t1, t2, t3, t4)
    # filter out to unique bytes, since bank conflicts will only occur from separate indices, different bank value
    t1_filtered = unique(t1);
    t2_filtered = unique(t2);
    t3_filtered = unique(t3);
    t4_filtered = unique(t4);

    # find bank indices
    t1_banks = [CalculateBankIndex(i) for i in t1_filtered];
    t2_banks = [CalculateBankIndex(i) for i in t2_filtered];
    t3_banks = [CalculateBankIndex(i) for i in t3_filtered];
    t4_banks = [CalculateBankIndex(i) for i in t4_filtered];

    # Find number of bank conflicts
    t1_bcs = CalculateNumberOfDuplicates(t1_banks);
    t2_bcs = CalculateNumberOfDuplicates(t2_banks);
    t3_bcs = CalculateNumberOfDuplicates(t3_banks);
    t4_bcs = CalculateNumberOfDuplicates(t4_banks);

    return t1_bcs, t2_bcs, t3_bcs, t4_bcs;
end

# Similar to Td3 of AES-128 in OpenSSL 0.9.7
SBOX = UInt32[
    0xf4a75051, 0x4165537e, 0x17a4c31a, 0x275e963a,
    0xab6bcb3b, 0x9d45f11f, 0xfa58abac, 0xe303934b,
    0x30fa5520, 0x766df6ad, 0xcc769188, 0x024c25f5,
    0xe5d7fc4f, 0x2acbd7c5, 0x35448026, 0x62a38fb5,
    0xb15a49de, 0xba1b6725, 0xea0e9845, 0xfec0e15d,
    0x2f7502c3, 0x4cf01281, 0x4697a38d, 0xd3f9c66b,
    0x8f5fe703, 0x929c9515, 0x6d7aebbf, 0x5259da95,
    0xbe832dd4, 0x7421d358, 0xe0692949, 0xc9c8448e,
    0xc2896a75, 0x8e7978f4, 0x583e6b99, 0xb971dd27,
    0xe14fb6be, 0x88ad17f0, 0x20ac66c9, 0xce3ab47d,
    0xdf4a1863, 0x1a3182e5, 0x51336097, 0x537f4562,
    0x6477e0b1, 0x6bae84bb, 0x81a01cfe, 0x082b94f9,
    0x48685870, 0x45fd198f, 0xde6c8794, 0x7bf8b752,
    0x73d323ab, 0x4b02e272, 0x1f8f57e3, 0x55ab2a66,
    0xeb2807b2, 0xb5c2032f, 0xc57b9a86, 0x3708a5d3,
    0x2887f230, 0xbfa5b223, 0x036aba02, 0x16825ced,
    0xcf1c2b8a, 0x79b492a7, 0x07f2f0f3, 0x69e2a14e,
    0xdaf4cd65, 0x05bed506, 0x34621fd1, 0xa6fe8ac4,
    0x2e539d34, 0xf355a0a2, 0x8ae13205, 0xf6eb75a4,
    0x83ec390b, 0x60efaa40, 0x719f065e, 0x6e1051bd,
    0x218af93e, 0xdd063d96, 0x3e05aedd, 0xe6bd464d,
    0x548db591, 0xc45d0571, 0x06d46f04, 0x5015ff60,
    0x98fb2419, 0xbde997d6, 0x4043cc89, 0xd99e7767,
    0xe842bdb0, 0x898b8807, 0x195b38e7, 0xc8eedb79,
    0x7c0a47a1, 0x420fe97c, 0x841ec9f8, 0x00000000,
    0x80868309, 0x2bed4832, 0x1170ac1e, 0x5a724e6c,
    0x0efffbfd, 0x8538560f, 0xaed51e3d, 0x2d392736,
    0x0fd9640a, 0x5ca62168, 0x5b54d19b, 0x362e3a24,
    0x0a67b10c, 0x57e70f93, 0xee96d2b4, 0x9b919e1b,
    0xc0c54f80, 0xdc20a261, 0x774b695a, 0x121a161c,
    0x93ba0ae2, 0xa02ae5c0, 0x22e0433c, 0x1b171d12,
    0x090d0b0e, 0x8bc7adf2, 0xb6a8b92d, 0x1ea9c814,
    0xf1198557, 0x75074caf, 0x99ddbbee, 0x7f60fda3,
    0x01269ff7, 0x72f5bc5c, 0x663bc544, 0xfb7e345b,
    0x4329768b, 0x23c6dccb, 0xedfc68b6, 0xe4f163b8,
    0x31dccad7, 0x63851042, 0x97224013, 0xc6112084,
    0x4a247d85, 0xbb3df8d2, 0xf93211ae, 0x29a16dc7,
    0x9e2f4b1d, 0xb230f3dc, 0x8652ec0d, 0xc1e3d077,
    0xb3166c2b, 0x70b999a9, 0x9448fa11, 0xe9642247,
    0xfc8cc4a8, 0xf03f1aa0, 0x7d2cd856, 0x3390ef22,
    0x494ec787, 0x38d1c1d9, 0xcaa2fe8c, 0xd40b3698,
    0xf581cfa6, 0x7ade28a5, 0xb78e26da, 0xadbfa43f,
    0x3a9de42c, 0x78920d50, 0x5fcc9b6a, 0x7e466254,
    0x8d13c2f6, 0xd8b8e890, 0x39f75e2e, 0xc3aff582,
    0x5d80be9f, 0xd0937c69, 0xd52da96f, 0x2512b3cf,
    0xac993bc8, 0x187da710, 0x9c636ee8, 0x3bbb7bdb,
    0x267809cd, 0x5918f46e, 0x9ab701ec, 0x4f9aa883,
    0x956e65e6, 0xffe67eaa, 0xbccf0821, 0x15e8e6ef,
    0xe79bd9ba, 0x6f36ce4a, 0x9f09d4ea, 0xb07cd629,
    0xa4b2af31, 0x3f23312a, 0xa59430c6, 0xa266c035,
    0x4ebc3774, 0x82caa6fc, 0x90d0b0e0, 0xa7d81533,
    0x04984af1, 0xecdaf741, 0xcd500e7f, 0x91f62f17,
    0x4dd68d76, 0xefb04d43, 0xaa4d54cc, 0x9604dfe4,
    0xd1b5e39e, 0x6a881b4c, 0x2c1fb8c1, 0x65517f46,
    0x5eea049d, 0x8c355d01, 0x877473fa, 0x0b412efb,
    0x671d5ab3, 0xdbd25292, 0x105633e9, 0xd647136d,
    0xd7618c9a, 0xa10c7a37, 0xf8148e59, 0x133c89eb,
    0xa927eece, 0x61c935b7, 0x1ce5ede1, 0x47b13c7a,
    0xd2df599c, 0xf2733f55, 0x14ce7918, 0xc737bf73,
    0xf7cdea53, 0xfdaa5b5f, 0x3d6f14df, 0x44db8678,
    0xaff381ca, 0x68c43eb9, 0x24342c38, 0xa3405fc2,
    0x1dc37216, 0xe2250cbc, 0x3c498b28, 0x0d9541ff,
    0xa8017139, 0x0cb3de08, 0xb4e49cd8, 0x56c19064,
    0xcb84617b, 0x32b670d5, 0x6c5c7448, 0xb85742d0,
];

# Generate Inverse SBOX
SBOX_INV = GenerateInverseSbox(SBOX);
# println(ConvertU32ToU8Array(0xb85742d0));

# Generate test data
test_data = rand(UInt32, NSAMPLES);

actual_key = 0x02358953; # looking pinpoint an individual byte

# Generate cipher texts and timing data
md = CuModuleFile(joinpath(@__DIR__, "sbox_encrypt.ptx"));
sbox_encrypt = CuFunction(md, "kernel_sbox_encrypt");

actual_timings = [];
actual_cipher_blocks = [];

for i in eachindex(test_data)
    CUDA.@sync begin
        cudacall(sbox_encrypt,
            Tuple{CuPtr{Cuint},CuPtr{Cuint},CuPtr{Cuint},CuPtr{Cuint},CuPtr{Cuint}}, 
            d_plaintext, d_ciphertext, d_key, d_sbox, d_runtime; threads=NTHREADS, blocks=NBLOCKS);
    end
end

# Calculate the actual average time for encryption from sample data + actual key

# For each cipher text collected, perform the following
# 1. For each possible key byte for the target lookup operation
# 2. Calculate the number of bank conflicts give that key byte guess for that instruction
# 3. Using the linear relationship from the benchmarks for bank conflicts, convert number of predicted conflicts to GPU time
# 4. Calculate the difference between the actual GPU time and predicted GPU time (difference of means)
# 5. Plot results