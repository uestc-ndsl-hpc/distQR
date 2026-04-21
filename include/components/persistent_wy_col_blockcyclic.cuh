#pragma once

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <vector>

namespace distributed_qr_col_blockcyclic {

enum class PersistentWyStorageMode {
    None = 0,
    Dense = 1,
    Compact = 2,
};

inline const char* PersistentWyStorageModeToString(PersistentWyStorageMode mode) {
    switch (mode) {
    case PersistentWyStorageMode::None:
        return "none";
    case PersistentWyStorageMode::Dense:
        return "dense";
    case PersistentWyStorageMode::Compact:
        return "compact";
    }
    return "unknown";
}

inline void PersistentWyAssertCuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        spdlog::error("{}: {}", context, cudaGetErrorString(status));
        std::exit(1);
    }
}

struct CompactWyBlockMeta {
    int block_begin = 0;
    int block_end = 0;
    int block_rows = 0;
    int kb = 0;
    size_t arena_offset = 0;
};

inline std::vector<CompactWyBlockMeta> BuildCompactWyBlockLayout(
    int m,
    int n,
    int nb,
    const std::vector<int>& block_starts,
    const std::vector<int>& block_ends,
    size_t* total_elems) {
    std::vector<CompactWyBlockMeta> metas;
    size_t arena_elems = 0;

    if (m <= 0 || n <= 0 || nb <= 0) {
        if (total_elems) {
            *total_elems = 0;
        }
        return metas;
    }

    for (size_t i = 0; i < block_starts.size(); ++i) {
        const int owner_block_begin = block_starts[i];
        const int owner_block_end = block_ends[i];
        for (int block_begin = owner_block_begin; block_begin < owner_block_end; block_begin += nb) {
            const int block_end = std::min(block_begin + nb, n);
            const int kb = block_end - block_begin;
            const int block_rows = m - block_begin;
            if (kb <= 0 || block_rows <= 0) {
                continue;
            }

            CompactWyBlockMeta meta{};
            meta.block_begin = block_begin;
            meta.block_end = block_end;
            meta.block_rows = block_rows;
            meta.kb = kb;
            meta.arena_offset = arena_elems;
            metas.push_back(meta);
            arena_elems += 2ULL * static_cast<size_t>(block_rows) * static_cast<size_t>(kb);
        }
    }

    if (total_elems) {
        *total_elems = arena_elems;
    }
    return metas;
}

inline size_t CompactWyStorageRequiredElems(int m,
                                            int n,
                                            int nb,
                                            const std::vector<int>& block_starts,
                                            const std::vector<int>& block_ends) {
    size_t total_elems = 0;
    static_cast<void>(BuildCompactWyBlockLayout(m, n, nb, block_starts, block_ends, &total_elems));
    return total_elems;
}

template <typename T>
struct PersistentWyStorage {
    PersistentWyStorageMode mode = PersistentWyStorageMode::None;

    T* d_w_dense = nullptr;
    T* d_y_dense = nullptr;
    int lda_local = 0;

    T* d_compact_arena = nullptr;
    size_t compact_arena_elems = 0;
    std::vector<CompactWyBlockMeta> compact_blocks;
};

template <typename T>
PersistentWyStorage<T> MakeNoPersistentWyStorage() {
    return PersistentWyStorage<T>{};
}

template <typename T>
PersistentWyStorage<T> MakeDensePersistentWyStorage(T* d_w_dense, T* d_y_dense, int lda_local) {
    PersistentWyStorage<T> storage{};
    if (!d_w_dense && !d_y_dense) {
        return storage;
    }

    storage.mode = PersistentWyStorageMode::Dense;
    storage.d_w_dense = d_w_dense;
    storage.d_y_dense = d_y_dense;
    storage.lda_local = lda_local;
    return storage;
}

template <typename T>
PersistentWyStorage<T> BuildCompactPersistentWyStorage(int m,
                                                       int n,
                                                       int nb,
                                                       const std::vector<int>& block_starts,
                                                       const std::vector<int>& block_ends,
                                                       T* d_compact_arena,
                                                       size_t compact_arena_elems) {
    PersistentWyStorage<T> storage{};
    storage.mode = PersistentWyStorageMode::Compact;
    storage.compact_blocks =
        BuildCompactWyBlockLayout(m, n, nb, block_starts, block_ends, &storage.compact_arena_elems);
    storage.d_compact_arena = d_compact_arena;

    if (storage.compact_arena_elems == 0) {
        return storage;
    }
    if (!d_compact_arena || compact_arena_elems < storage.compact_arena_elems) {
        spdlog::error(
            "Compact persistent WY storage too small (need {} elems, got {}; ptr={}).",
            storage.compact_arena_elems, compact_arena_elems,
            static_cast<const void*>(d_compact_arena));
        std::exit(1);
    }
    return storage;
}

template <typename T>
bool PersistentWyHasAnyStorage(const PersistentWyStorage<T>& storage) {
    if (storage.mode == PersistentWyStorageMode::None) {
        return false;
    }
    if (storage.mode == PersistentWyStorageMode::Dense) {
        return storage.d_w_dense || storage.d_y_dense;
    }
    return storage.d_compact_arena || storage.compact_blocks.empty();
}

template <typename T>
bool PersistentWyHasCompleteFactors(const PersistentWyStorage<T>& storage) {
    if (storage.mode == PersistentWyStorageMode::Dense) {
        return storage.d_w_dense && storage.d_y_dense;
    }
    if (storage.mode == PersistentWyStorageMode::Compact) {
        return storage.d_compact_arena || storage.compact_blocks.empty();
    }
    return false;
}

template <typename T>
const CompactWyBlockMeta* FindCompactWyBlock(const PersistentWyStorage<T>& storage, int block_begin) {
    if (storage.mode != PersistentWyStorageMode::Compact) {
        return nullptr;
    }

    for (const auto& meta : storage.compact_blocks) {
        if (meta.block_begin == block_begin) {
            return &meta;
        }
    }
    return nullptr;
}

template <typename T>
T* CompactWyWPtr(const PersistentWyStorage<T>& storage, const CompactWyBlockMeta& meta) {
    return storage.d_compact_arena + meta.arena_offset;
}

template <typename T>
T* CompactWyYPtr(const PersistentWyStorage<T>& storage, const CompactWyBlockMeta& meta) {
    return CompactWyWPtr(storage, meta) +
           static_cast<size_t>(meta.block_rows) * static_cast<size_t>(meta.kb);
}

template <typename T>
void StorePersistentWyBlock(const PersistentWyStorage<T>& storage,
                            int block_begin,
                            int local_block_col,
                            int block_rows,
                            int kb,
                            const T* src_w,
                            int ld_src_w,
                            const T* src_y,
                            int ld_src_y,
                            cudaStream_t stream) {
    if (storage.mode == PersistentWyStorageMode::None || block_rows <= 0 || kb <= 0) {
        return;
    }

    if (storage.mode == PersistentWyStorageMode::Dense) {
        if ((storage.d_w_dense || storage.d_y_dense) && (storage.lda_local <= 0 || local_block_col < 0)) {
            spdlog::error("Dense persistent WY store got invalid lda/local_block_col (lda={} local_col={}).",
                          storage.lda_local, local_block_col);
            std::exit(1);
        }
        if (storage.d_w_dense) {
            PersistentWyAssertCuda(
                cudaMemcpy2DAsync(storage.d_w_dense +
                                      static_cast<size_t>(local_block_col) * storage.lda_local +
                                      block_begin,
                                  static_cast<size_t>(storage.lda_local) * sizeof(T), src_w,
                                  static_cast<size_t>(ld_src_w) * sizeof(T),
                                  static_cast<size_t>(block_rows) * sizeof(T), kb,
                                  cudaMemcpyDeviceToDevice, stream),
                "cudaMemcpy2DAsync persistent dense W store");
        }
        if (storage.d_y_dense) {
            PersistentWyAssertCuda(
                cudaMemcpy2DAsync(storage.d_y_dense +
                                      static_cast<size_t>(local_block_col) * storage.lda_local +
                                      block_begin,
                                  static_cast<size_t>(storage.lda_local) * sizeof(T), src_y,
                                  static_cast<size_t>(ld_src_y) * sizeof(T),
                                  static_cast<size_t>(block_rows) * sizeof(T), kb,
                                  cudaMemcpyDeviceToDevice, stream),
                "cudaMemcpy2DAsync persistent dense Y store");
        }
        return;
    }

    const CompactWyBlockMeta* meta = FindCompactWyBlock(storage, block_begin);
    if (!meta) {
        spdlog::error("Compact persistent WY block {} not found during store.", block_begin);
        std::exit(1);
    }
    if (meta->block_rows != block_rows || meta->kb != kb) {
        spdlog::error(
            "Compact persistent WY block {} shape mismatch during store (expected rows={} kb={}, got rows={} kb={}).",
            block_begin, meta->block_rows, meta->kb, block_rows, kb);
        std::exit(1);
    }

    PersistentWyAssertCuda(
        cudaMemcpy2DAsync(CompactWyWPtr(storage, *meta),
                          static_cast<size_t>(meta->block_rows) * sizeof(T), src_w,
                          static_cast<size_t>(ld_src_w) * sizeof(T),
                          static_cast<size_t>(meta->block_rows) * sizeof(T), meta->kb,
                          cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpy2DAsync persistent compact W store");
    PersistentWyAssertCuda(
        cudaMemcpy2DAsync(CompactWyYPtr(storage, *meta),
                          static_cast<size_t>(meta->block_rows) * sizeof(T), src_y,
                          static_cast<size_t>(ld_src_y) * sizeof(T),
                          static_cast<size_t>(meta->block_rows) * sizeof(T), meta->kb,
                          cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpy2DAsync persistent compact Y store");
}

template <typename T>
void LoadPersistentWyBlock(const PersistentWyStorage<T>& storage,
                           int block_begin,
                           int block_rows,
                           int kb,
                           int local_block_col,
                           T* dst_w,
                           T* dst_y,
                           int ld_dst,
                           cudaStream_t stream) {
    if (storage.mode == PersistentWyStorageMode::None) {
        spdlog::error("LoadPersistentWyBlock requires stored WY, but mode is none.");
        std::exit(1);
    }
    if (!dst_w || !dst_y || block_rows <= 0 || kb <= 0 || ld_dst < block_rows) {
        spdlog::error(
            "LoadPersistentWyBlock got invalid destination shape (rows={} kb={} ld_dst={}).",
            block_rows, kb, ld_dst);
        std::exit(1);
    }

    if (storage.mode == PersistentWyStorageMode::Dense) {
        if (!storage.d_w_dense || !storage.d_y_dense || storage.lda_local <= 0 || local_block_col < 0) {
            spdlog::error(
                "Dense persistent WY load got invalid storage (W={} Y={} lda={} local_col={}).",
                static_cast<const void*>(storage.d_w_dense), static_cast<const void*>(storage.d_y_dense),
                storage.lda_local, local_block_col);
            std::exit(1);
        }
        PersistentWyAssertCuda(
            cudaMemcpy2DAsync(dst_w, static_cast<size_t>(ld_dst) * sizeof(T),
                              storage.d_w_dense +
                                  static_cast<size_t>(local_block_col) * storage.lda_local +
                                  block_begin,
                              static_cast<size_t>(storage.lda_local) * sizeof(T),
                              static_cast<size_t>(block_rows) * sizeof(T), kb,
                              cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpy2DAsync persistent dense W load");
        PersistentWyAssertCuda(
            cudaMemcpy2DAsync(dst_y, static_cast<size_t>(ld_dst) * sizeof(T),
                              storage.d_y_dense +
                                  static_cast<size_t>(local_block_col) * storage.lda_local +
                                  block_begin,
                              static_cast<size_t>(storage.lda_local) * sizeof(T),
                              static_cast<size_t>(block_rows) * sizeof(T), kb,
                              cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpy2DAsync persistent dense Y load");
        return;
    }

    const CompactWyBlockMeta* meta = FindCompactWyBlock(storage, block_begin);
    if (!meta) {
        spdlog::error("Compact persistent WY block {} not found during load.", block_begin);
        std::exit(1);
    }
    if (meta->block_rows != block_rows || meta->kb != kb || ld_dst < block_rows) {
        spdlog::error(
            "Compact persistent WY block {} load shape mismatch (expected rows={} kb={}, got rows={} kb={} ld_dst={}).",
            block_begin, meta->block_rows, meta->kb, block_rows, kb, ld_dst);
        std::exit(1);
    }

    PersistentWyAssertCuda(
        cudaMemcpy2DAsync(dst_w, static_cast<size_t>(ld_dst) * sizeof(T),
                          CompactWyWPtr(storage, *meta),
                          static_cast<size_t>(meta->block_rows) * sizeof(T),
                          static_cast<size_t>(block_rows) * sizeof(T), kb,
                          cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpy2DAsync persistent compact W load");
    PersistentWyAssertCuda(
        cudaMemcpy2DAsync(dst_y, static_cast<size_t>(ld_dst) * sizeof(T),
                          CompactWyYPtr(storage, *meta),
                          static_cast<size_t>(meta->block_rows) * sizeof(T),
                          static_cast<size_t>(block_rows) * sizeof(T), kb,
                          cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpy2DAsync persistent compact Y load");
}

}  // namespace distributed_qr_col_blockcyclic
