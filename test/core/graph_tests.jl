using Test
using AMDGPU
using GPUArrays

@testset "HIP Graphs" begin
    @testset "+1" begin
        f!(o) = o .+= one(eltype(o))

        z = AMDGPU.zeros(Int, 4, 4)
        graph = AMDGPU.@captured f!(z)
        @test sum(z) == 16

        AMDGPU.launch(graph)
        @test sum(z) == 16 * 2
        AMDGPU.launch(graph)
        @test sum(z) == 16 * 3
    end

    @testset "malloc/free" begin
        function f!(o)
            x = AMDGPU.ones(eltype(o), size(o))
            o .+= x .+ one(eltype(o))
            AMDGPU.unsafe_free!(x)
        end

        z = AMDGPU.zeros(Int, 4, 4)
        graph = AMDGPU.@captured f!(z)
        @test sum(z) == 32

        AMDGPU.launch(graph)
        @test sum(z) == 32 * 2
        AMDGPU.launch(graph)
        @test sum(z) == 32 * 3
    end

    @testset "only malloc + alloc cache" begin
        function f!(o)
            x = AMDGPU.ones(eltype(o), size(o))
            y = AMDGPU.ones(eltype(o), size(o))
            o .+= (x * y) .+ one(eltype(o))
        end

        z = AMDGPU.zeros(Int, 4, 4)
        cache = GPUArrays.AllocCache()
        # Pre-populate alloc cache, to avoid malloc calls during capture.
        GPUArrays.@cached cache f!(z)
        # Capture with alloc cache.
        graph = GPUArrays.@cached cache AMDGPU.@captured f!(z)
        @test sum(z) == length(z) * 5 * 2

        AMDGPU.launch(graph)
        @test sum(z) == length(z) * 5 * 3
        AMDGPU.launch(graph)
        @test sum(z) == length(z) * 5 * 4
    end

    # A capture records whatever reaches its stream, from any thread, and a
    # finalizer freeing an array the GC collected is such work. Recorded, the free
    # is a graph node for memory the graph never allocated, and the first
    # `hipGraphLaunch` segfaulted after `hsa_amd_vmem_unmap failed`. Surfaced
    # through Mantle, which records with the split `begincapture!`/`endcapture!`
    # and so had none of the `GC.enable(false)` that `capture` does.
    @testset "a free of older memory waits for the capture to close" begin
        function nodetypes(g)
            n = Ref{Csize_t}(0)
            AMDGPU.HIP.hipGraphGetNodes(g, C_NULL, n)
            nodes = Vector{AMDGPU.HIP.hipGraphNode_t}(undef, n[])
            AMDGPU.HIP.hipGraphGetNodes(g, nodes, n)
            t = Ref{AMDGPU.HIP.hipGraphNodeType}()
            return map(nodes) do node
                AMDGPU.HIP.hipGraphNodeGetType(node, t)
                t[]
            end
        end

        s = AMDGPU.stream()
        z = AMDGPU.zeros(Int, 16)
        z .+= 1                     # compiled here: compiling inside a capture invalidates it
        here = AMDGPU.zeros(Int, 1 << 20)
        elsewhere = AMDGPU.zeros(Int, 1 << 20)
        AMDGPU.synchronize()
        stats = AMDGPU.memory_stats(AMDGPU.device())
        before = stats.live

        AMDGPU.HIP.begincapture!(s)
        z .+= 1
        # What the GC does to an unreachable array, on this thread and another.
        finalize(here)
        wait(Threads.@spawn finalize(elsewhere))
        during = stats.live
        z .+= 1
        g = AMDGPU.HIP.endcapture!(s)

        @test nodetypes(g) == [AMDGPU.HIP.hipGraphNodeTypeKernel,
                               AMDGPU.HIP.hipGraphNodeTypeKernel]
        # Held back while the capture was open, issued when it closed. `<=`
        # because a collection inside the window defers its frees here too.
        @test during == before
        @test stats.live <= before - 2 * sizeof(Int) * (1 << 20)
        exec = AMDGPU.HIP.instantiate(g)
        AMDGPU.HIP.launch(exec, s)
        AMDGPU.HIP.launch(exec, s)
        @test Array(z) == fill(5, 16)
    end

    @testset "Update graph" begin
        f1!(o) = o .+= one(eltype(o))
        f2!(o) = o .+= eltype(o)(2)

        z = AMDGPU.zeros(Int, 4, 4)
        graph = AMDGPU.@captured f1!(z)
        @test sum(z) == 16

        g_new = AMDGPU.capture() do
            f2!(z)
        end
        @test AMDGPU.update(graph, g_new)
        AMDGPU.launch(graph)
        @test sum(z) == 16 * 3
    end
end
