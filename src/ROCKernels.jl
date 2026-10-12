module ROCKernels

export ROCBackend

import AMDGPU
import AMDGPU.Device: @device_override
using AMDGPU: GPUArrays, rocSPARSE, HIP, Device, rocconvert, hipfunction

import Adapt
import KernelInterface as KI
import Core: LLVMPtr, VecElement

"""
    ROCBackend <: KernelInterface.Backend

KernelInterface backend that executes kernels on an AMD GPU via AMDGPU.jl.
Pass `ROCBackend()` to a KernelAbstractions kernel to run it on the GPU, or
obtain it from an array with `KernelInterface.get_backend(::ROCArray)`.

Printing from a kernel (`KernelAbstractions.@print`) is not supported: it does nothing.
"""
struct ROCBackend <: KI.Backend end

KI.functional(::ROCBackend) = AMDGPU.functional()
KI.versioninfo(io::IO, ::ROCBackend) = AMDGPU.versioninfo(io)
KI.ndevices(::ROCBackend) = AMDGPU.HIP.ndevices()
KI.device(::ROCBackend) = AMDGPU.device_id()
function KI.device!(kab::ROCBackend, id::Int)
    (0 < id <= KI.ndevices(kab)) || throw(ArgumentError("Device id $id out of bounds."))
    AMDGPU.device_id!(id)
    return
end
KI.device(::ROCBackend, A::AMDGPU.ROCArray) = AMDGPU.device_id(AMDGPU.device(A))

Adapt.adapt_storage(::ROCBackend, a::AbstractArray) = Adapt.adapt(AMDGPU.ROCArray, a)
Adapt.adapt_storage(::ROCBackend, a::Union{AMDGPU.ROCArray, GPUArrays.AbstractGPUSparseArray}) = a

KI.get_backend(::AMDGPU.ROCArray) = ROCBackend()
KI.get_backend(::AMDGPU.rocSPARSE.ROCSparseVector) = ROCBackend()
KI.get_backend(::AMDGPU.rocSPARSE.ROCSparseMatrixCSC) = ROCBackend()
KI.get_backend(::AMDGPU.rocSPARSE.ROCSparseMatrixCSR) = ROCBackend()

KI.synchronize(::ROCBackend) = AMDGPU.synchronize()

# an event recorded on, and waited for by, the task's stream
KI.record_event(::ROCBackend) = HIP.HIPEvent(AMDGPU.stream())
function KI.wait_event(::ROCBackend, ev::HIP.HIPEvent)
    HIP.hipStreamWaitEvent(AMDGPU.stream(), ev, 0)
    return
end

KI.supports_float64(::ROCBackend) = true
KI.supports_atomics(::ROCBackend) = true
KI.supports_subgroups(::ROCBackend) = true
# `shfl_down` decomposes other types into 32-bit shuffles
KI.supports_shuffle(::ROCBackend, ::Type{T}) where {T} =
    T <: Union{Bool, Base.BitInteger, Base.IEEEFloat, Complex{<:Union{Base.BitInteger, Base.IEEEFloat}}}
# `shfl` decomposes like `shfl_down`
KI.shfl_types(::ROCBackend) = DataType[Bool,
                                       UInt8, UInt16, UInt32, UInt64, UInt128,
                                       Int8, Int16, Int32, Int64, Int128,
                                       Float16, Float32, Float64,
                                       ComplexF16, ComplexF32, ComplexF64]
# the element types of `__ockl_wfred_add`, the wavefront reduction every lane gets the
# sum of
const REDUCE_ADD_TYPES = (Int32, UInt32, Int64, UInt64, Float16, Float32, Float64)
KI.sub_group_reduce_add_types(::ROCBackend) = DataType[REDUCE_ADD_TYPES...]

function KI.priority!(::ROCBackend, priority::Symbol)
    priority ∉ (:high, :normal, :low) && error(
        "Priority `$priority` must be one of `:high`, `:normal`, `:low`.")
    AMDGPU.priority!(priority)
    return
end

## memory operations

KI.unsafe_free!(x::AMDGPU.ROCArray) = AMDGPU.unsafe_free!(x)
KI.allocate(::ROCBackend, ::Type{T}, dims::Tuple) where T = AMDGPU.ROCArray{T}(undef, dims)

# dense arrays, and contiguous views of them
const ContiguousArray{T} = Union{DenseArray{T}, Base.FastContiguousSubArray{T, <:Any, <:DenseArray}}
on_device(A::ContiguousArray) = parent(A) isa AMDGPU.ROCArray

function KI.copyto!(::ROCBackend, A::ContiguousArray{T}, B::ContiguousArray{T}) where T
    length(A) == length(B) ||
        throw(ArgumentError("Arrays must have the same length, got $(length(A)) and $(length(B))"))
    if isbitstype(T) && (on_device(A) || on_device(B))
        # queued on the task's stream, after the work queued before it
        GC.@preserve A B begin
            AMDGPU.Mem.memcpy!(pointer(A), pointer(B), length(A) * AMDGPU.aligned_sizeof(T);
                               stream=AMDGPU.stream())
        end
    else
        # host-to-host copies, and bits unions, whose type tags are stored separately.
        # queued work may still access host arrays (e.g. a copy from the device), so wait
        # for it first.
        on_device(A) || on_device(B) || AMDGPU.synchronize()
        copyto!(A, B)
    end
    return A
end
KI.copyto!(::ROCBackend, A, B) =
    throw(ArgumentError("KernelInterface.copyto! only supports contiguous arrays of the same element type, got $(typeof(A)) and $(typeof(B))"))

function KI.pagelock!(::ROCBackend, x::Array)
    AMDGPU.Mem.pin(pointer(x), sizeof(x))
    return
end

## kernel launch

KI.argconvert(::ROCBackend, arg) = rocconvert(arg)

# a compiled kernel, and the callable it was compiled from. the kernel only holds pointers
# to the arrays the callable captures, so the callable has to be kept alive.
struct ROCKernel{F, K <: AMDGPU.Runtime.HIPKernel}
    f::F
    kernel::K
end

function KI.kernel_function(backend::ROCBackend, f::F, tt::TT=Tuple{}; name=nothing, kwargs...) where {F,TT}
    # kernels have to execute with the device's wavefront size, which is what `hipfunction`
    # compiles for by default
    ws = HIP.wavefrontsize(AMDGPU.device())
    if haskey(kwargs, :wavefrontsize64) && kwargs[:wavefrontsize64] != (ws == 64)
        throw(ArgumentError("`wavefrontsize64=$(kwargs[:wavefrontsize64])` conflicts with the wavefront size of the device, $ws"))
    end
    kernel = hipfunction(rocconvert(f), tt; name, kwargs...)
    return KI.Kernel(backend, ROCKernel(f, kernel))
end

function KI.launch(obj::KI.Kernel{ROCBackend}, groups::Dims{3}, items::Dims{3},
                   args::Tuple; kwargs...)
    # KernelInterface has validated the launch geometry
    if haskey(kwargs, :groupsize) || haskey(kwargs, :gridsize)
        throw(ArgumentError("KernelInterface kernels take `numgroups`, `workgroupsize` or `ndrange`, not `groupsize` or `gridsize`"))
    end
    f = obj.kern.f
    kernel = obj.kern.kernel
    stream = get(kwargs, :stream, AMDGPU.stream())
    GC.@preserve f begin
        # convert the callable again, like the arguments, which makes the arrays it captures
        # available to the stream
        kernel = typeof(kernel)(rocconvert(f, stream), kernel.fun)
        AMDGPU.Runtime.launch_tuple(kernel, args; groupsize=items, gridsize=groups, kwargs..., stream)
    end
    return
end

function KI.max_work_group_size(kernel::KI.Kernel{ROCBackend})::Int
    max_items = Ref{Cint}()
    HIP.hipFuncGetAttribute(max_items, HIP.HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel.kern.kernel.fun)
    return Int(max_items[])
end
function KI.launch_configuration(kernel::KI.Kernel{ROCBackend}; nitems::Union{Integer,Nothing}=nothing,
                                 max_work_group_size::Integer=typemax(Int))
    max_items = min(max_work_group_size, something(nitems, typemax(Int)),
                    KI.max_work_group_size(kernel))
    (; groupsize) = AMDGPU.launch_configuration(kernel.kern.kernel; max_block_size=max_items)
    return (; workgroupsize=Int(min(groupsize, max_items)))
end
function KI.max_work_group_size(::ROCBackend)::Int
    Int(HIP.attribute(AMDGPU.device(), HIP.hipDeviceAttributeMaxThreadsPerBlock))
end
# queried on every automatically-sized launch, so use the limits cached in the device
KI.max_work_group_dims(::ROCBackend)::NTuple{3, Int} = HIP.max_workgroup_dims(AMDGPU.device())
# HIP takes the grid size in workgroups, but the dispatch packet holds it in work-items
# (as a UInt32 per dimension, which HIP checks), and the device code assumes workgroup
# indices fit in an Int32 (see `Device._max_groups`). Report the number of workgroups
# that can be launched with any valid workgroup size. HIP's `maxGridSize` isn't usable:
# depending on the ROCm version it holds CUDA's block limits or the work-item limits.
function KI.max_num_groups(backend::ROCBackend)::NTuple{3, Int}
    dims = KI.max_work_group_dims(backend)
    return ntuple(Val(3)) do d
        Int(min(Device._max_groups[d], Device._max_grid_size[d] ÷ dims[d]))
    end
end
# `KI.kernel_function` compiles for the device's wavefront size
function KI.sub_group_size(::ROCBackend)::Int
    Int(HIP.wavefrontsize(AMDGPU.device()))
end
function KI.multiprocessor_count(::ROCBackend)::Int
    Int(HIP.attribute(AMDGPU.device(), HIP.hipDeviceAttributeMultiprocessorCount))
end

## COV_EXCL_START

## indexing

# computed with `% T`, which unlike `T(x)` has no error path

@device_override @inline function KI.get_local_id(::Type{T}) where {T}
    return (; x = Device.workitemIdx().x % T, y = Device.workitemIdx().y % T, z = Device.workitemIdx().z % T)
end

@device_override @inline function KI.get_group_id(::Type{T}) where {T}
    return (; x = Device.workgroupIdx().x % T, y = Device.workgroupIdx().y % T, z = Device.workgroupIdx().z % T)
end

@device_override @inline function KI.get_local_size(::Type{T}) where {T}
    return (; x = Device.workgroupDim().x % T, y = Device.workgroupDim().y % T, z = Device.workgroupDim().z % T)
end

@device_override @inline function KI.get_num_groups(::Type{T}) where {T}
    return (; x = Device.gridGroupDim().x % T, y = Device.gridGroupDim().y % T, z = Device.gridGroupDim().z % T)
end

## sub-groups

# wavefronts are formed from consecutive linear work-item indices
@inline function linear_workitem_id()
    return (Device.workitemIdx().x - 0x1) +
           (Device.workitemIdx().y - 0x1) * Device.workgroupDim().x +
           (Device.workitemIdx().z - 0x1) * Device.workgroupDim().x * Device.workgroupDim().y
end

@inline workgroup_items() = Device.workgroupDim().x * Device.workgroupDim().y * Device.workgroupDim().z

# the 0-based lane of the work-item in its wavefront. unlike `Device.activelane()`, this
# doesn't depend on which lanes are active. `mbcnt.hi` adds nothing in wave32 mode.
@inline function hardware_lane()
    lo = ccall("llvm.amdgcn.mbcnt.lo", llvmcall, UInt32, (UInt32, UInt32), typemax(UInt32), 0x0)
    return ccall("llvm.amdgcn.mbcnt.hi", llvmcall, UInt32, (UInt32, UInt32), typemax(UInt32), lo)
end

# the last wavefront of a workgroup can be partial
@device_override @inline function KI.get_sub_group_size(::Type{T}) where {T}
    ws = Device.wavefrontsize()
    return min(ws, workgroup_items() - (linear_workitem_id() ÷ ws) * ws) % T
end

@device_override KI.get_max_sub_group_size(::Type{T}) where {T} = Device.wavefrontsize() % T

@device_override KI.get_num_sub_groups(::Type{T}) where {T} = cld(workgroup_items(), Device.wavefrontsize()) % T

@device_override KI.get_sub_group_id(::Type{T}) where {T} = (linear_workitem_id() ÷ Device.wavefrontsize() + 0x1) % T

@device_override KI.get_sub_group_local_id(::Type{T}) where {T} = (hardware_lane() + 0x1) % T

## shared memory

@device_override @inline function KI.localmemory(::Type{T}, ::Val{Dims}) where {T, Dims}
    # every call site gets its own memory, see `alloc_special`
    ptr = Device.alloc_special(Val(:localmemory), T, Val(AMDGPU.AS.Local), Val(prod(Dims)))
    AMDGPU.ROCDeviceArray(Dims, ptr)
end

## cooperative matrices
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

## synchronization and printing

@device_override @inline function KI.barrier()
    Device.sync_workgroup()
end

@device_override @inline function KI.sub_group_barrier()
    Device.sync_wavefront()
end

@device_override function KI.shfl_down(val::T, offset::Integer) where T
    @inline Device.shfl_down(val, offset % Cint)
end

@device_override function KI.shfl(val::T, lane::Integer) where T
    @inline Device.shfl(val, lane % Cint)
end

# the hardware reduction rather than a `shfl_down` loop, which is why KernelInterface
# keeps it a primitive
@device_override @inline KI.sub_group_reduce_add(val::Union{REDUCE_ADD_TYPES...}) =
    Device.wfred(+, val)

# not supported, see the `ROCBackend` docstring
@device_override @inline KI._print(args...) = nothing

## COV_EXCL_STOP

end
