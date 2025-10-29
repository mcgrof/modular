// ============================================================================
// AMR Benchmark - C++ Reference Implementation
// Direct comparison with Mojo version to measure compiler optimizations
// ============================================================================

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>

// Structure-of-Arrays mesh for fair comparison with Mojo
struct AMRMesh {
    int capacity;
    int num_cells;
    int num_active;
    int nx;
    int ny;

    // SoA fields
    std::vector<int> ids;
    std::vector<int> levels;
    std::vector<float> x_coords;
    std::vector<float> y_coords;
    std::vector<float> cell_sizes;
    std::vector<float> temperatures;
    std::vector<float> gradients;
    std::vector<int> active_flags;

    // CSR neighbor connectivity
    std::vector<int> neighbor_offsets;
    std::vector<int> neighbor_ids;

    AMRMesh(int cap, int nx_, int ny_)
        : capacity(cap), num_cells(0), num_active(0), nx(nx_), ny(ny_) {
        ids.resize(capacity);
        levels.resize(capacity);
        x_coords.resize(capacity);
        y_coords.resize(capacity);
        cell_sizes.resize(capacity);
        temperatures.resize(capacity);
        gradients.resize(capacity);
        active_flags.resize(capacity);
        neighbor_offsets.resize(capacity + 1);
        neighbor_ids.resize(capacity * 8);
    }
};

void initialize_uniform_mesh(AMRMesh& mesh, int nx, int ny, bool build_csr) {
    int cell_id = 0;
    float dx = 1.0f / nx;
    float dy = 1.0f / ny;

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int idx = cell_id;
            mesh.ids[idx] = cell_id;
            mesh.levels[idx] = 0;
            mesh.x_coords[idx] = (i + 0.5f) * dx;
            mesh.y_coords[idx] = (j + 0.5f) * dy;
            mesh.cell_sizes[idx] = dx;
            mesh.temperatures[idx] = 0.0f;
            mesh.gradients[idx] = 0.0f;
            mesh.active_flags[idx] = 1;
            cell_id++;
        }
    }

    mesh.num_cells = nx * ny;
    mesh.num_active = nx * ny;

    // Build CSR graph only for unstructured
    if (build_csr) {
        int edge_count = 0;
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int cell_idx = j * nx + i;
                mesh.neighbor_offsets[cell_idx] = edge_count;

                if (i > 0) {  // Left
                    mesh.neighbor_ids[edge_count++] = cell_idx - 1;
                }
                if (i < nx - 1) {  // Right
                    mesh.neighbor_ids[edge_count++] = cell_idx + 1;
                }
                if (j > 0) {  // Bottom
                    mesh.neighbor_ids[edge_count++] = cell_idx - nx;
                }
                if (j < ny - 1) {  // Top
                    mesh.neighbor_ids[edge_count++] = cell_idx + nx;
                }
            }
        }
        mesh.neighbor_offsets[nx * ny] = edge_count;
    }
}

void set_gaussian_initial_condition(AMRMesh& mesh) {
    for (int i = 0; i < mesh.num_cells; i++) {
        float x = mesh.x_coords[i];
        float y = mesh.y_coords[i];
        float dx = x - 0.5f;
        float dy = y - 0.5f;
        float r2 = dx * dx + dy * dy;
        float sigma_sq = 0.01f;
        mesh.temperatures[i] = (r2 < sigma_sq) ? 100.0f : 0.0f;
    }
}

// Structured mesh: direct neighbor addressing
void heat_diffusion_structured(AMRMesh& mesh, float dt, float alpha) {
    std::vector<float> temp_new(mesh.num_cells);

    #pragma omp parallel for
    for (int idx = 0; idx < mesh.num_cells; idx++) {
        if (mesh.active_flags[idx] == 0) {
            temp_new[idx] = mesh.temperatures[idx];
            continue;
        }

        int i = idx % mesh.nx;
        int j = idx / mesh.nx;
        float temp_center = mesh.temperatures[idx];
        float laplacian = 0.0f;
        int neighbor_count = 0;

        // Direct addressing - compiler can optimize
        if (i > 0) {
            laplacian += mesh.temperatures[idx - 1] - temp_center;
            neighbor_count++;
        }
        if (i < mesh.nx - 1) {
            laplacian += mesh.temperatures[idx + 1] - temp_center;
            neighbor_count++;
        }
        if (j > 0) {
            laplacian += mesh.temperatures[idx - mesh.nx] - temp_center;
            neighbor_count++;
        }
        if (j < mesh.ny - 1) {
            laplacian += mesh.temperatures[idx + mesh.nx] - temp_center;
            neighbor_count++;
        }

        float dx2 = mesh.cell_sizes[idx] * mesh.cell_sizes[idx];
        if (neighbor_count > 0) {
            temp_new[idx] = temp_center + dt * alpha * laplacian / dx2;
        } else {
            temp_new[idx] = temp_center;
        }
    }

    // Copy back
    for (int i = 0; i < mesh.num_cells; i++) {
        mesh.temperatures[i] = temp_new[i];
    }
}

// Unstructured mesh: indirect CSR addressing
void heat_diffusion_unstructured(AMRMesh& mesh, float dt, float alpha) {
    std::vector<float> temp_new(mesh.num_cells);

    #pragma omp parallel for
    for (int i = 0; i < mesh.num_cells; i++) {
        if (mesh.active_flags[i] == 0) {
            temp_new[i] = mesh.temperatures[i];
            continue;
        }

        float temp_center = mesh.temperatures[i];
        float laplacian = 0.0f;
        int neighbor_count = 0;

        int start = mesh.neighbor_offsets[i];
        int end = mesh.neighbor_offsets[i + 1];

        // Double indirection - challenging for traditional compilers
        for (int k = start; k < end; k++) {
            int nbr_id = mesh.neighbor_ids[k];  // First indirection
            laplacian += mesh.temperatures[nbr_id] - temp_center;  // Second indirection
            neighbor_count++;
        }

        float dx2 = mesh.cell_sizes[i] * mesh.cell_sizes[i];
        if (neighbor_count > 0) {
            temp_new[i] = temp_center + dt * alpha * laplacian / dx2;
        } else {
            temp_new[i] = temp_center;
        }
    }

    // Copy back
    for (int i = 0; i < mesh.num_cells; i++) {
        mesh.temperatures[i] = temp_new[i];
    }
}

double benchmark_structured(int num_steps, int nx, int ny) {
    AMRMesh mesh(nx * ny * 2, nx, ny);
    initialize_uniform_mesh(mesh, nx, ny, false);
    set_gaussian_initial_condition(mesh);

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < num_steps; step++) {
        heat_diffusion_structured(mesh, 0.0001f, 1.0f);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    return elapsed.count();
}

double benchmark_unstructured(int num_steps, int nx, int ny) {
    AMRMesh mesh(nx * ny * 2, nx, ny);
    initialize_uniform_mesh(mesh, nx, ny, true);  // Build CSR graph
    set_gaussian_initial_condition(mesh);

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < num_steps; step++) {
        heat_diffusion_unstructured(mesh, 0.0001f, 1.0f);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    return elapsed.count();
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "AMR Benchmark: C++ Reference Implementation\n";
    std::cout << "======================================================================\n\n";

    const int BENCH_NX = 64;
    const int BENCH_NY = 64;
    const int BENCH_STEPS = 100;

    std::cout << "Configuration:\n";
    std::cout << "  Grid: " << BENCH_NX << " x " << BENCH_NY << " = "
              << BENCH_NX * BENCH_NY << " cells\n";
    std::cout << "  Timesteps: " << BENCH_STEPS << "\n";
    std::cout << "  Compiler: " << __VERSION__ << "\n\n";

    // Benchmark structured mesh
    std::cout << "[1] Benchmarking STRUCTURED mesh (direct addressing)...\n";
    double time_structured = benchmark_structured(BENCH_STEPS, BENCH_NX, BENCH_NY);
    double throughput_s = (BENCH_NX * BENCH_NY * BENCH_STEPS) / time_structured / 1e6;
    std::cout << "    Time: " << std::fixed << std::setprecision(6)
              << time_structured << " seconds\n";
    std::cout << "    Throughput: " << std::fixed << std::setprecision(2)
              << throughput_s << " Mcells/sec\n\n";

    // Benchmark unstructured mesh
    std::cout << "[2] Benchmarking UNSTRUCTURED mesh (indirect CSR addressing)...\n";
    double time_unstructured = benchmark_unstructured(BENCH_STEPS, BENCH_NX, BENCH_NY);
    double throughput_u = (BENCH_NX * BENCH_NY * BENCH_STEPS) / time_unstructured / 1e6;
    std::cout << "    Time: " << std::fixed << std::setprecision(6)
              << time_unstructured << " seconds\n";
    std::cout << "    Throughput: " << std::fixed << std::setprecision(2)
              << throughput_u << " Mcells/sec\n\n";

    // Performance analysis
    double overhead_percent = (time_unstructured / time_structured - 1.0) * 100.0;
    double ratio = time_unstructured / time_structured;

    std::cout << "[3] Performance Analysis:\n";
    std::cout << "    Indirect access overhead: " << std::fixed << std::setprecision(2)
              << overhead_percent << " %\n";
    std::cout << "    Unstructured/Structured ratio: " << std::fixed << std::setprecision(2)
              << ratio << " x\n";

    std::cout << "\n======================================================================\n";
    std::cout << "C++ Compiler Performance:\n";
    std::cout << "======================================================================\n";
    std::cout << "  Direct addressing:   " << std::fixed << std::setprecision(2)
              << throughput_s << " Mcells/sec\n";
    std::cout << "  Indirect CSR access: " << std::fixed << std::setprecision(2)
              << throughput_u << " Mcells/sec (" << overhead_percent << "% overhead)\n";
    std::cout << "======================================================================\n";

    return 0;
}
