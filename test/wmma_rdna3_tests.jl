using Test
using AMDGPU
using AMDGPU: ROCArray, @roc
using BFloat16s
using AMDGPU.Device: WMMA_RDNA3, workitemIdx, workgroupIdx
import KernelInterface as KI

AMDGPU.allowscalar(false)

# Only run WMMA_RDNA3 tests on RDNA3+ (gfx1100+)
_arch_str = first(split(AMDGPU.HIP.gcn_arch(AMDGPU.device()), ':'))
gfx = parse(Int, _arch_str[4:end])
is_rdna3 = 1100 ≤ gfx < 1200
is_rdna4 = 1200 ≤ gfx < 1300
if !is_rdna3 && !is_rdna4
    @info "Skipping WMMA_RDNA3 tests (requires RDNA3+)"
else
    # Tile base pointer + stride for A (M×K) by layout.
    _a_tile(ptr, ::Type{WMMA_RDNA3.ColMajor}, tile_row, k, M, K, ::Type{T}) where T =
        ptr + (k * M + tile_row) * Int32(sizeof(T)), M
    _a_tile(ptr, ::Type{WMMA_RDNA3.RowMajor}, tile_row, k, M, K, ::Type{T}) where T =
        ptr + (tile_row * K + k) * Int32(sizeof(T)), K

    # Tile base pointer + stride for B (K×N) by layout.
    _b_tile(ptr, ::Type{WMMA_RDNA3.ColMajor}, tile_col, k, N, K, ::Type{T}) where T =
        ptr + (tile_col * K + k) * Int32(sizeof(T)), K
    _b_tile(ptr, ::Type{WMMA_RDNA3.RowMajor}, tile_col, k, N, K, ::Type{T}) where T =
        ptr + (k * N + tile_col) * Int32(sizeof(T)), N

    function wmma_kernel!(
        C::AbstractArray{R},
        A::AbstractArray{T}, B,
        M::Int32, N::Int32, K::Int32,
        layout, scale::Float32,
    ) where {R, T}
        tile_row = (workgroupIdx().x - Int32(1)) * Int32(WMMA_RDNA3.M)
        tile_col = (workgroupIdx().y - Int32(1)) * Int32(WMMA_RDNA3.N)

        C_ptr = pointer(C)
        A_ptr = pointer(A)
        B_ptr = pointer(B)

        c_frag = WMMA_RDNA3.fill_c(Float32, 0f0)
        k = Int32(0)
        while k < K
            a_ptr, a_stride = _a_tile(A_ptr, layout, tile_row, k, M, K, T)
            b_ptr, b_stride = _b_tile(B_ptr, layout, tile_col, k, N, K, T)

            a_frag = WMMA_RDNA3.load_a(a_ptr, a_stride, layout)
            b_frag = WMMA_RDNA3.load_b(b_ptr, b_stride, layout)
            c_frag = WMMA_RDNA3.mma(a_frag, b_frag, c_frag)

            k += Int32(WMMA_RDNA3.K)
        end

        c_frag = c_frag .* scale
        c_ptr = C_ptr + (tile_col * M + tile_row) * Int32(sizeof(R))
        WMMA_RDNA3.store_d(c_ptr, c_frag, M, WMMA_RDNA3.ColMajor)
        return
    end

    function ki_coopmat_kernel!(C, A, B)
        MA = KI.CoopMatrix{Float16,16,16,KI.MatrixA,KI.SubgroupScope}
        MB = KI.CoopMatrix{Float16,16,16,KI.MatrixB,KI.SubgroupScope}
        MC = KI.CoopMatrix{Float32,16,16,KI.Accumulator,KI.SubgroupScope}

        # This is the path fused kernels actually use: stage through local
        # memory, whose backend-neutral spelling becomes a ROCDeviceArray over
        # an addrspace(3) pointer on AMDGPU.
        sa = KI.localmemory(Float16, Val((256,)), Val(:ki_coopmat_a))
        sb = KI.localmemory(Float16, Val((256,)), Val(:ki_coopmat_b))
        lane = Int(workitemIdx().x)
        for i in lane:32:256
            sa[i] = A[i]
            sb[i] = B[i]
        end
        KI.barrier()

        a = KI.coopmat_load(MA, sa, 1, 16, Val(false))
        b = KI.coopmat_load(MB, sb, 1, 16, Val(false))
        c = KI.coopmat_muladd(a, b, KI.coopmat_zero(MC))

        # Exercise the portable component-wise and component-access floor too:
        # these are used by fused epilogues and attention rescaling, not merely
        # conveniences around the WMMA instruction itself.
        c = KI.coopmat_add(KI.coopmat_mul(c, c), c)
        for i in Int32(0):(KI.coopmat_length(MC) - Int32(1))
            c = KI.coopmat_setcomp(c, i, KI.coopmat_getcomp(c, i) + 1f0)
        end
        KI.coopmat_store(pointer(C), 1, 16, c, Val(false))
        return
    end

    function ki_accumulator_conversion_kernel!(loaded, rounded, half_input, float_input)
        MH = KI.CoopMatrix{Float16,16,16,KI.Accumulator,KI.SubgroupScope}
        MF = KI.CoopMatrix{Float32,16,16,KI.Accumulator,KI.SubgroupScope}

        # Fused epilogues use accumulator loads for fp16 bias tiles, then
        # convert the fp32 product to logical fp16 before applying activation.
        # Both operations keep a native fp32 WMMA fragment underneath on RDNA.
        h = KI.coopmat_load(MH, pointer(half_input), 1, 16, Val(false))
        KI.coopmat_store(pointer(loaded), 1, 16, KI.coopmat_convert(MF, h), Val(false))

        f = KI.coopmat_load(MF, pointer(float_input), 1, 16, Val(false))
        KI.coopmat_store(pointer(rounded), 1, 16, KI.coopmat_convert(MH, f), Val(false))
        return
    end

    @testset "WMMA_RDNA3" begin
        @testset "KernelInterface cooperative-matrix adapter" begin
            A_host = rand(Float16, 16, 16)
            B_host = rand(Float16, 16, 16)
            A, B = ROCArray(A_host), ROCArray(B_host)
            C = ROCArray(zeros(Float32, 16, 16))

            @roc gridsize=32 groupsize=32 ki_coopmat_kernel!(C, A, B)
            product = Float32.(A_host) * Float32.(B_host)
            expected = product .* product .+ product .+ 1f0
            @test maximum(abs.(Array(C) .- expected)) < 0.01

            half_host = rand(Float16, 16, 16)
            # Deliberately use values between adjacent fp16 numbers so this
            # catches a conversion that only relabels the native fp32 handle.
            float_host = rand(Float32, 16, 16) .* 4f0 .- 2f0
            half_input = ROCArray(half_host)
            float_input = ROCArray(float_host)
            loaded = ROCArray(zeros(Float32, 16, 16))
            rounded = similar(loaded)

            @roc gridsize=32 groupsize=32 ki_accumulator_conversion_kernel!(
                loaded, rounded, half_input, float_input)
            @test Array(loaded) == Float32.(half_host)
            @test Array(rounded) == Float32.(Float16.(float_host))
        end

        @testset "ColMajor $M×$N: $arg_T -> $res_T" for (M, N, K) in (
            (64, 64, 64), (128, 128, 128),
        ), arg_T in (Float16, BFloat16), res_T in (Float16, BFloat16, Float32)
            A_host = arg_T.(rand(M, K))
            B_host = arg_T.(rand(K, N))
            A, B = ROCArray(A_host), ROCArray(B_host)
            C = ROCArray(zeros(res_T, M, N))
            tol = sizeof(res_T) == 4 ? 0.1 : 0.3

            tiles_m, tiles_n = M ÷ WMMA_RDNA3.M, N ÷ WMMA_RDNA3.N
            @roc gridsize=(tiles_m, tiles_n) groupsize=32 wmma_kernel!(
                C, A, B, Int32(M), Int32(N), Int32(K), WMMA_RDNA3.ColMajor, 1f0)
            @test maximum(abs.(Float32.(C) .- (Float32.(A) * Float32.(B)))) < tol
        end

        @testset "Fragment broadcast: $M×$N $arg_T" for (M, N, K) in (
            (64, 64, 64), (128, 128, 128),
        ), arg_T in (Float16, BFloat16)
            scale = rand(Float32)
            A_host = arg_T.(rand(M, K))
            B_host = arg_T.(rand(K, N))
            A, B = ROCArray(A_host), ROCArray(B_host)
            C = ROCArray(zeros(Float32, M, N))

            tiles_m, tiles_n = M ÷ WMMA_RDNA3.M, N ÷ WMMA_RDNA3.N
            @roc gridsize=(tiles_m, tiles_n) groupsize=32 wmma_kernel!(
                C, A, B, Int32(M), Int32(N), Int32(K), WMMA_RDNA3.ColMajor, scale)
            expected = scale .* (Float32.(A) * Float32.(B))
            @test maximum(abs.(Float32.(C) .- expected)) < 0.1
        end

        @testset "RowMajor $M×$N: $arg_T" for (M, N, K) in (
            (64, 64, 64), (128, 128, 128),
        ), arg_T in (Float16, BFloat16), res_T in (Float16, BFloat16, Float32)
            A_host = arg_T.(rand(M, K))
            B_host = arg_T.(rand(K, N))
            # Transpose to get row-major storage.
            A = ROCArray(A_host')
            B = ROCArray(B_host')
            C = ROCArray(zeros(res_T, M, N))
            tol = sizeof(res_T) == 4 ? 0.1 : 0.3

            tiles_m, tiles_n = M ÷ WMMA_RDNA3.M, N ÷ WMMA_RDNA3.N
            @roc gridsize=(tiles_m, tiles_n) groupsize=32 wmma_kernel!(
                C, A, B, Int32(M), Int32(N), Int32(K), WMMA_RDNA3.RowMajor, 1f0)
            @test maximum(abs.(
                Float32.(C) .- ROCArray(Float32.(A_host)) * ROCArray(Float32.(B_host))
            )) < tol
        end
    end
end
