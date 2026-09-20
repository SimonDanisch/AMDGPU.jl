module ROCInterface

export ROCBackend

import AMDGPU
import AMDGPU: rocconvert, hipfunction
import AMDGPU.Device: @device_override
using AMDGPU: GPUArrays, rocSPARSE, HIP, Device

import Adapt
import KernelInterface as KI
import LLVM
import Core: LLVMPtr, VecElement

using StaticArraysCore: MArray

"""
    ROCBackend <: KernelAbstractions.GPU

KernelAbstractions backend that executes kernels on an AMD GPU via AMDGPU.jl.
Pass `ROCBackend()` to a KernelAbstractions kernel to run it on the GPU, or
obtain it from an array with `KernelAbstractions.get_backend(::ROCArray)`.
"""
struct ROCBackend <: KI.GPU end

KI.versioninfo(io::IO, ::ROCBackend) = AMDGPU.versioninfo(io)

KI.functional(::ROCBackend) = AMDGPU.functional()
KI.ndevices(::ROCBackend) = AMDGPU.HIP.ndevices()
KI.device(::ROCBackend) = AMDGPU.device_id()
function KI.device!(kab::ROCBackend, id::Int)
    (0 < id <= KI.ndevices(kab)) || throw(ArgumentError("Device id $id out of bounds."))
    AMDGPU.device_id!(id)
    return
end

Adapt.adapt_storage(::ROCBackend, a::AbstractArray) = Adapt.adapt(AMDGPU.ROCArray, a)
Adapt.adapt_storage(::ROCBackend, a::Union{AMDGPU.ROCArray, GPUArrays.AbstractGPUSparseArray}) = a

KI.get_backend(::AMDGPU.ROCArray) = ROCBackend()
KI.get_backend(::AMDGPU.rocSPARSE.ROCSparseVector) = ROCBackend()
KI.get_backend(::AMDGPU.rocSPARSE.ROCSparseMatrixCSC) = ROCBackend()
KI.get_backend(::AMDGPU.rocSPARSE.ROCSparseMatrixCSR) = ROCBackend()

KI.synchronize(::ROCBackend) = AMDGPU.synchronize()

KI.unsafe_free!(x::AMDGPU.ROCArray) = AMDGPU.unsafe_free!(x)
KI.allocate(::ROCBackend, ::Type{T}, dims::Tuple) where T = AMDGPU.ROCArray{T}(undef, dims)
KI.zeros(::ROCBackend, ::Type{T}, dims::Tuple) where T = AMDGPU.zeros(T, dims)
KI.ones(::ROCBackend, ::Type{T}, dims::Tuple) where T = AMDGPU.ones(T, dims)

function KI.priority!(::ROCBackend, priority::Symbol)
    priority ∉ (:high, :normal, :low) && error(
        "Priority `$priority` must be one of `:high`, `:normal`, `:low`.")
    AMDGPU.priority!(priority)
end

function KI.copyto!(::ROCBackend, A, B)
    GC.@preserve A B begin
        copyto!(A, 1, B, 1, length(A))
    end
    return
end

function KI.pagelock!(::ROCBackend, x::Array)
    AMDGPU.Mem.pin(pointer(x), sizeof(x))
    return
end

KI.argconvert(::ROCBackend, arg) = rocconvert(arg)

function KI.kernel_function(::ROCBackend, f::F, tt::TT=Tuple{}; name=nothing, kwargs...) where {F,TT}
    kern = hipfunction(f, tt; name, kwargs...)
    KI.Kernel{ROCBackend, typeof(kern)}(ROCBackend(), kern)
end

function (obj::KI.Kernel{ROCBackend})(args...; numworkgroups=(), workgroupsize=(), ndrange=(), max_work_group_size=typemax(Int))
    KI.check_launch_args(numworkgroups, workgroupsize, ndrange)
    prod(ndrange) == 0 && return nothing

    numworkgroups, workgroupsize = KI.auto_launch_sizes(obj, numworkgroups, workgroupsize, ndrange, max_work_group_size)
    obj.kern(args...; groupsize = workgroupsize, gridsize = numworkgroups)
    return nothing
end

function KI.kernel_max_work_group_size(kikern::KI.Kernel{<:ROCBackend}; max_work_items::Int=Int(typemax(Int32)))::Int
    (; groupsize) = AMDGPU.launch_configuration(kikern.kern; max_block_size = max_work_items)

    return Int(min(max_work_items, groupsize))
end
function KI.max_work_group_size(::ROCBackend)::Int
    Int(HIP.attribute(AMDGPU.HIP.device(), AMDGPU.HIP.hipDeviceAttributeMaxThreadsPerBlock))
end
function KI.sub_group_size(::ROCBackend)::Int
    HIP.wavefrontsize(HIP.device())
end
function KI.multiprocessor_count(::ROCBackend)::Int
    Int(HIP.attribute(AMDGPU.HIP.device(), AMDGPU.HIP.hipDeviceAttributeMultiprocessorCount))
end

KI.shfl_down_types(::ROCBackend) = DataType[Bool,
                                             UInt8, UInt16, UInt32, UInt64, UInt128,
                                             Int8, Int16, Int32, Int64, Int128,
                                             Float16, Float32, Float64,
                                             ComplexF16, ComplexF32, ComplexF64]

# Indexing.
## COV_EXCL_START
@device_override @inline function KI.get_local_id()
    return (; x = Int(AMDGPU.Device.workitemIdx().x), y = Int(AMDGPU.Device.workitemIdx().y), z = Int(AMDGPU.Device.workitemIdx().z))
end

@device_override @inline function KI.get_group_id()
    return (; x = Int(AMDGPU.Device.workgroupIdx().x), y = Int(AMDGPU.Device.workgroupIdx().y), z = Int(AMDGPU.Device.workgroupIdx().z))
end

@device_override @inline function KI.get_global_id()
    return (; x = Int((AMDGPU.Device.workgroupIdx().x-1)*AMDGPU.Device.blockDim().x + AMDGPU.Device.workitemIdx().x), y = Int((AMDGPU.Device.workgroupIdx().y-1)*AMDGPU.Device.blockDim().y + AMDGPU.Device.workitemIdx().y), z = Int((AMDGPU.Device.workgroupIdx().z-1)*AMDGPU.Device.blockDim().z + AMDGPU.Device.workitemIdx().z))
end

@device_override @inline function KI.get_local_size()
    return (; x = Int(AMDGPU.Device.workgroupDim().x), y = Int(AMDGPU.Device.workgroupDim().y), z = Int(AMDGPU.Device.workgroupDim().z))
end

@device_override @inline function KI.get_num_groups()
    return (; x = Int(AMDGPU.Device.gridGroupDim().x), y = Int(AMDGPU.Device.gridGroupDim().y), z = Int(AMDGPU.Device.gridGroupDim().z))
end

@device_override @inline function KI.get_global_size()
    return (; x = Int(AMDGPU.Device.gridItemDim().x), y = Int(AMDGPU.Device.gridItemDim().y), z = Int(AMDGPU.Device.gridItemDim().z))
end

@device_override KI.get_sub_group_size() = UInt32(Device.wavefrontsize())

@device_override KI.get_max_sub_group_size() = UInt32(Device.wavefrontsize())

@device_override KI.get_num_sub_groups() = UInt32(prod(Device.blockDim()) ÷ Device.wavefrontsize())

@device_override KI.get_sub_group_id() = UInt32(((Device.threadIdx().x - 1) + Device.blockDim().x * (Device.threadIdx().y - 1) + Device.blockDim().x * Device.blockDim().y * (Device.threadIdx().z - 1)) ÷ Device.wavefrontsize()) + 0x1

@device_override KI.get_sub_group_local_id() = UInt32(Device.activelane() + 0x1)

# Shared memory.

# `Val(Id)` and not `Val(:shmem)`. `alloc_special` names its global
# `alloc_special_$id`, so a constant id made every workgroup buffer in a kernel
# the same buffer: two `KI.localmemory(Float32, (16, 16))` calls are one
# generated function, one global, and a tiled kernel that stages two tiles gets
# the second on top of the first. `KA.SharedMemory` above has always passed its
# `@localmem` id here; this is the same fix on the KI side.
@device_override @inline function KI.localmemory(::Type{T}, ::Val{Dims}, ::Val{Id}) where {T, Dims, Id}
    ptr = AMDGPU.Device.alloc_special(Val(Id), T, Val(AMDGPU.AS.Local), Val(prod(Dims)))
    AMDGPU.ROCDeviceArray(Dims, ptr)
end

# Cooperative matrices -------------------------------------------------------
#
# KernelInterface deliberately hides the backend storage parameter of
# `CoopMatrix`.  On AMDGPU it is the native RDNA3 WMMA register fragment, not
# Lava's Int32 SPIR-V SSA token.  Keeping the wrapper is important: portable
# kernels continue to dispatch on MatrixA/MatrixB/Accumulator while the value
# carried through LLVM is AMDGPU's real register tuple.
const _WMMA3 = Device.WMMA_RDNA3
const _WMMA3AB = _WMMA3.Fragment{16,16,Float16,16}
const _WMMA3C = _WMMA3.Fragment{16,16,Float32,8}

@inline _wmma3_layout(::Val{false}) = _WMMA3.ColMajor
@inline _wmma3_layout(::Val{true}) = _WMMA3.RowMajor
@inline _wmma3_ptr(ptr::LLVMPtr{T,A}, offset::Integer) where {T,A} =
    ptr + Int32(offset - 1) * Int32(sizeof(T))
@inline _wmma3_scalarptr(
    ptr::LLVMPtr{NTuple{W,VecElement{Float16}},A},
) where {W,A} = reinterpret(LLVMPtr{Float16,A}, ptr)

# `@localmem` is a `ROCDeviceArray` around an addrspace(3) pointer.  Keep that
# wrapper out of the matrix implementation just as Lava keeps its shared-array
# wrapper out of the SPIR-V lowering: unwrap once, then use the same pointer
# primitive for local and global memory.
@device_override @inline KI.coopmat_load(
    mt::Type{<:KI.CoopMatrix}, src::Device.ROCDeviceArray,
    offset::Integer, stride::Integer,
) = KI.coopmat_load(mt, pointer(src), offset, stride)

@device_override @inline KI.coopmat_load(
    mt::Type{<:KI.CoopMatrix}, src::Device.ROCDeviceArray,
    offset::Integer, stride::Integer, layout::Val,
) = KI.coopmat_load(mt, pointer(src), offset, stride, layout)

# The portable staged GEMM vectorises both its global loads and its shared
# storage.  Its offsets and strides are consequently counted in W-wide shared
# elements.  RDNA's WMMA loader takes scalar fp16 addresses, so reinterpret the
# packed storage and convert those two units exactly once at this boundary.
@device_override @inline KI.coopmat_load(
    mt::Type{<:KI.CoopMatrix},
    src::Device.ROCDeviceArray{NTuple{W,VecElement{Float16}},N,A},
    offset::Integer, stride::Integer,
) where {W,N,A} = KI.coopmat_load(
    mt, _wmma3_scalarptr(pointer(src)), 1 + (offset - 1) * W, stride * W)

@device_override @inline KI.coopmat_load(
    mt::Type{<:KI.CoopMatrix},
    src::Device.ROCDeviceArray{NTuple{W,VecElement{Float16}},N,A},
    offset::Integer, stride::Integer, layout::Val,
) where {W,N,A} = KI.coopmat_load(
    mt, _wmma3_scalarptr(pointer(src)), 1 + (offset - 1) * W, stride * W, layout)

@device_override @inline KI.coopmat_store(
    dst::Device.ROCDeviceArray, offset::Integer, stride::Integer,
    m::KI.CoopMatrix, layout::Val=Val(false),
) = KI.coopmat_store(pointer(dst), offset, stride, m, layout)

@device_override @inline function KI.coopmat_load(
    ::Type{KI.CoopMatrix{Float16,16,16,KI.MatrixA,KI.SubgroupScope}},
    ptr::LLVMPtr{Float16,A}, offset::Integer, stride::Integer,
    layout::Val{RM}=Val(false),
) where {A,RM}
    f = _WMMA3.load_a(_wmma3_ptr(ptr, offset), Int32(stride), _wmma3_layout(layout))
    return KI.CoopMatrix{Float16,16,16,KI.MatrixA,KI.SubgroupScope}(f)
end

@device_override @inline function KI.coopmat_load(
    ::Type{KI.CoopMatrix{Float16,16,16,KI.MatrixB,KI.SubgroupScope}},
    ptr::LLVMPtr{Float16,A}, offset::Integer, stride::Integer,
    layout::Val{RM}=Val(false),
) where {A,RM}
    f = _WMMA3.load_b(_wmma3_ptr(ptr, offset), Int32(stride), _wmma3_layout(layout))
    return KI.CoopMatrix{Float16,16,16,KI.MatrixB,KI.SubgroupScope}(f)
end

@device_override @inline function KI.coopmat_load(
    ::Type{KI.CoopMatrix{T,16,16,KI.Accumulator,KI.SubgroupScope}},
    ptr::LLVMPtr{S,A}, offset::Integer, stride::Integer,
    layout::Val{RM}=Val(false),
) where {T<:Union{Float16,Float32},S<:Union{Float16,Float32},A,RM}
    f = _WMMA3.load_c(_wmma3_ptr(ptr, offset), Int32(stride), _wmma3_layout(layout))
    return KI.CoopMatrix{T,16,16,KI.Accumulator,KI.SubgroupScope}(f)
end

@device_override @inline function KI.coopmat_store(
    ptr::LLVMPtr{T,A}, offset::Integer, stride::Integer,
    m::KI.CoopMatrix{S,16,16,KI.Accumulator,KI.SubgroupScope,_WMMA3C},
    layout::Val{RM}=Val(false),
) where {T<:Union{Float16,Float32},A,S,RM}
    _WMMA3.store_d(_wmma3_ptr(ptr, offset), m.handle, Int32(stride),
                   _wmma3_layout(layout))
    return nothing
end

@device_override @inline function KI.coopmat_muladd(
    a::KI.CoopMatrix{Float16,16,16,KI.MatrixA,KI.SubgroupScope,_WMMA3AB},
    b::KI.CoopMatrix{Float16,16,16,KI.MatrixB,KI.SubgroupScope,_WMMA3AB},
    c::KI.CoopMatrix{Float32,16,16,KI.Accumulator,KI.SubgroupScope,_WMMA3C},
)
    f = _WMMA3.mma(a.handle, b.handle, c.handle)
    return KI.CoopMatrix{Float32,16,16,KI.Accumulator,KI.SubgroupScope}(f)
end

@device_override @inline function KI.coopmat_mul(
    a::KI.CoopMatrix{T,16,16,U,KI.SubgroupScope,_WMMA3C},
    b::KI.CoopMatrix{T,16,16,U,KI.SubgroupScope,_WMMA3C},
) where {T,U}
    return KI.CoopMatrix{T,16,16,U,KI.SubgroupScope}(a.handle .* b.handle)
end

@device_override @inline function KI.coopmat_mul(
    a::KI.CoopMatrix{Float16,16,16,U,KI.SubgroupScope,_WMMA3AB},
    b::KI.CoopMatrix{Float16,16,16,U,KI.SubgroupScope,_WMMA3AB},
) where {U<:Union{KI.MatrixA,KI.MatrixB}}
    return KI.CoopMatrix{Float16,16,16,U,KI.SubgroupScope}(a.handle .* b.handle)
end

@device_override @inline function KI.coopmat_add(
    a::KI.CoopMatrix{T,16,16,U,KI.SubgroupScope,_WMMA3C},
    b::KI.CoopMatrix{T,16,16,U,KI.SubgroupScope,_WMMA3C},
) where {T,U}
    return KI.CoopMatrix{T,16,16,U,KI.SubgroupScope}(a.handle .+ b.handle)
end

@device_override @inline function KI.coopmat_add(
    a::KI.CoopMatrix{Float16,16,16,U,KI.SubgroupScope,_WMMA3AB},
    b::KI.CoopMatrix{Float16,16,16,U,KI.SubgroupScope,_WMMA3AB},
) where {U<:Union{KI.MatrixA,KI.MatrixB}}
    return KI.CoopMatrix{Float16,16,16,U,KI.SubgroupScope}(a.handle .+ b.handle)
end

@inline _wmma3_zero_ab() =
    _WMMA3AB(ntuple(_ -> VecElement(Float16(0)), Val(16)))

@device_override @inline KI.coopmat_zero(
    ::Type{KI.CoopMatrix{Float16,16,16,U,KI.SubgroupScope}},
) where {U<:Union{KI.MatrixA,KI.MatrixB}} =
    KI.CoopMatrix{Float16,16,16,U,KI.SubgroupScope}(_wmma3_zero_ab())

@device_override @inline KI.coopmat_zero(
    ::Type{KI.CoopMatrix{Float32,16,16,KI.Accumulator,KI.SubgroupScope}},
) = KI.CoopMatrix{Float32,16,16,KI.Accumulator,KI.SubgroupScope}(
        _WMMA3.fill_c(Float32, 0.0f0))

# An undefined fragment has no observable contents before a complete tensor
# load.  AMDGPU has no Julia-level undef vector value, so use zero; this changes
# no legal program and avoids manufacturing poison through a tuple constructor.
@device_override @inline KI.coopmat_undef(
    ::Type{KI.CoopMatrix{T,16,16,U,KI.SubgroupScope}},
) where {T,U} = KI.coopmat_zero(KI.CoopMatrix{T,16,16,U,KI.SubgroupScope})

@device_override @inline KI.coopmat_length(
    ::Type{KI.CoopMatrix{T,16,16,KI.Accumulator,KI.SubgroupScope}},
) where {T<:Union{Float16,Float32}} = Int32(8)

@device_override @inline KI.coopmat_length(
    ::Type{KI.CoopMatrix{Float16,16,16,U,KI.SubgroupScope}},
) where {U<:Union{KI.MatrixA,KI.MatrixB}} = Int32(16)

@device_override @inline KI.coopmat_getcomp(
    m::KI.CoopMatrix{T,16,16,U,KI.SubgroupScope,_WMMA3C}, i::Int32,
) where {T,U} = m.handle.data[Int(i) + 1].value

@device_override @inline KI.coopmat_getcomp(
    m::KI.CoopMatrix{Float16,16,16,U,KI.SubgroupScope,_WMMA3AB}, i::Int32,
) where {U<:Union{KI.MatrixA,KI.MatrixB}} = m.handle.data[Int(i) + 1].value

@device_override @inline function KI.coopmat_setcomp(
    m::KI.CoopMatrix{T,16,16,U,KI.SubgroupScope,_WMMA3C}, i::Int32, v::Float32,
) where {T<:Union{Float16,Float32},U}
    j = Int(i) + 1
    data = ntuple(Val(8)) do k
        # The native accumulator is physically fp32 even when the portable
        # matrix is logically fp16.  Preserve that representation, but preserve
        # the portable operation's rounding too: fused GEMM activations run
        # after the graph's fp32 -> fp16 conversion.
        x = T === Float16 ? Float32(Float16(v)) : v
        k == j ? VecElement(x) : m.handle.data[k]
    end
    return KI.CoopMatrix{T,16,16,U,KI.SubgroupScope}(_WMMA3C(data))
end


@device_override @inline function KI.coopmat_setcomp(
    m::KI.CoopMatrix{Float16,16,16,U,KI.SubgroupScope,_WMMA3AB},
    i::Int32, v::Float16,
) where {U<:Union{KI.MatrixA,KI.MatrixB}}
    j = Int(i) + 1
    data = ntuple(Val(16)) do k
        k == j ? VecElement(v) : m.handle.data[k]
    end
    return KI.CoopMatrix{Float16,16,16,U,KI.SubgroupScope}(_WMMA3AB(data))
end

@device_override @inline KI.coopmat_convert(
    ::Type{KI.CoopMatrix{T,16,16,KI.Accumulator,KI.SubgroupScope}},
    m::KI.CoopMatrix{S,16,16,KI.Accumulator,KI.SubgroupScope,_WMMA3C},
) where {T<:Union{Float16,Float32},S<:Union{Float16,Float32}} = begin
    # RDNA stores every accumulator through its fp32 fragment.  A logical fp16
    # conversion therefore has to round the fragment explicitly; leaving that
    # until `store_d` is equivalent for identity, but wrong when portable code
    # applies an activation between `convert` and the store.
    data = T === Float16 ?
        ntuple(i -> VecElement(Float32(Float16(m.handle.data[i].value))), Val(8)) :
        m.handle.data
    KI.CoopMatrix{T,16,16,KI.Accumulator,KI.SubgroupScope}(_WMMA3C(data))
end

# Other.

@device_override @inline function KI.barrier()
    AMDGPU.Device.sync_workgroup()
end

@device_override @inline function KI.sub_group_barrier()
    AMDGPU.Device.sync_wavefront()
end

@device_override function KI.shfl_down(val::T, offset::Integer) where T
    @inline AMDGPU.Device.shfl_down(val, Cint(offset))
end

@device_override @inline function KI._print(args...)
    # TODO
end
## COV_EXCL_STOP

end
