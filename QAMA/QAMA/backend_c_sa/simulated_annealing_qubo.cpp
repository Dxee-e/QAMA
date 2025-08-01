#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <chrono>

torch::Tensor simulated_annealing_qubo(const torch::Tensor& Q,
                                       const float initial_temperature,
                                       const float alpha,
                                       const float cutoff_temperature,
                                       const int iterations_per_t,
                                       const int patience,
                                       const bool is_local_search,
                                       const bool flag_evolution_history,
                                       const std::string& history_file_path,
                                       const c10::optional<int64_t> rand_seed,
                                       const bool is_input_check,
                                       const c10::optional<torch::Tensor> csa_init_x) {
    if (is_input_check) {
        TORCH_CHECK(Q.dim() == 3, "Input tensor Q must be 3D (with batch dimension).");
        TORCH_CHECK(Q.size(2) == Q.size(1), "Input tensor Q must be a square matrix.");
        TORCH_CHECK(torch::allclose(Q, Q.transpose(-1, -2), 1e-7, 1e-7), "Input tensor Q must be a symmetric matrix.")
    }

    torch::Device device = Q.device();
    if (rand_seed.has_value()) {
        torch::manual_seed(rand_seed.value());
    }

    int batch_size = Q.size(0);
    int n = Q.size(1);

    torch::Tensor ha_history_tensor;
    if (flag_evolution_history) {
        ha_history_tensor = torch::empty({0, batch_size}, torch::kFloat32).to(device);
    }

    // Initialize solutions randomly or with provided initial solutions
    torch::Tensor current_solutions;
    if (csa_init_x.has_value()) {
        torch::Tensor init_x = csa_init_x.value();
        if (is_input_check) {
            TORCH_CHECK(init_x.dim() == 2, "csa_init_x must be a 2D tensor.");
            TORCH_CHECK(init_x.size(0) == batch_size, "csa_init_x must have the same batch size as Q.");
            TORCH_CHECK(init_x.size(1) == n, "csa_init_x must have the same number of variables as Q.");
        }
        if (init_x.device() != device) {
            init_x = init_x.to(device);
        }
        if (init_x.dtype() != torch::kBool) {
            init_x = init_x.to(torch::kBool);
        }
        current_solutions = init_x.clone();
    }
    else {
        current_solutions = torch::bernoulli(torch::full({batch_size, n}, 0.5, torch::kFloat32)).to(torch::kBool).to(device);
    }
    
    torch::Tensor temperatures = torch::full({batch_size}, initial_temperature, torch::kFloat32).to(device);

    torch::Tensor batch_indices = torch::arange(batch_size).to(device);
    torch::Tensor full_x = torch::ones({batch_size, n}, torch::kBool).to(device);
    torch::Tensor full_energy_2D = Q.clone();
    torch::Tensor full_energy_diag = full_energy_2D.diagonal(0, 1, 2);
    full_energy_2D = full_energy_2D * 2 - torch::diag_embed(full_energy_diag, 0, 1, 2);

    int no_improvement_count = 0;
    torch::Tensor x_mask = (current_solutions.unsqueeze(2) * current_solutions.unsqueeze(1)).to(torch::kBool);

    torch::Tensor best_energies = torch::sum(x_mask * Q, {1, 2});
    torch::Tensor best_solutions = current_solutions.clone();

    while ((temperatures > cutoff_temperature).any().item<bool>() && no_improvement_count < patience) {
        for (int i = 0; i < iterations_per_t; i++) {
            torch::Tensor idxs = torch::randint(0, n, {batch_size}).to(device);
            torch::Tensor mask = torch::zeros_like(current_solutions);
            mask.scatter_(1, idxs.unsqueeze(1), 1);
            torch::Tensor new_solutions = current_solutions.clone();
            new_solutions = torch::logical_and(  // XOR
                torch::logical_or(new_solutions, mask), torch::logical_not(torch::logical_and(new_solutions, mask)));

            torch::Tensor selected_values = new_solutions.index({batch_indices, idxs});
            torch::Tensor new_x_mask = x_mask.clone();
            torch::Tensor update_x_mask_vector = selected_values.unsqueeze(1) * new_solutions;
            new_x_mask.index_put_({batch_indices, idxs}, update_x_mask_vector);
            new_x_mask.index_put_({batch_indices, torch::indexing::Ellipsis, idxs}, update_x_mask_vector);
            torch::Tensor delta_sign = selected_values * 2 - 1;

            torch::Tensor row_new_x_mask = new_x_mask.index({batch_indices, idxs});
            torch::Tensor row_x_mask = x_mask.index({batch_indices, idxs});
            torch::Tensor row_chaged_x_mask = torch::logical_and(  // XOR
                torch::logical_or(row_new_x_mask, row_x_mask),
                torch::logical_not(torch::logical_and(row_new_x_mask, row_x_mask)));
            torch::Tensor delta_energies =
                delta_sign * torch::sum(row_chaged_x_mask * full_energy_2D.index({batch_indices, idxs}), {1});

            torch::Tensor acceptance_probabilities = torch::exp(-delta_energies / temperatures);
            torch::Tensor random_values = torch::rand_like(delta_energies);
            torch::Tensor accept_mask = (delta_energies < 0) | (acceptance_probabilities > random_values);
            current_solutions = torch::where(accept_mask.unsqueeze(1), new_solutions, current_solutions);

            x_mask = torch::where(accept_mask.unsqueeze(1).unsqueeze(2), new_x_mask, x_mask);
        }
        temperatures *= alpha;

        torch::Tensor current_energies = torch::sum(x_mask * Q, {1, 2});
        if (flag_evolution_history) {
            ha_history_tensor = torch::cat({ha_history_tensor, current_energies.unsqueeze(0)}, 0);
        }
        torch::Tensor improved_mask = current_energies < best_energies;
        if (improved_mask.any().item<bool>()) {
            best_energies = torch::where(improved_mask, current_energies, best_energies);
            best_solutions = torch::where(improved_mask.unsqueeze(1), current_solutions, best_solutions);
            no_improvement_count = 0;
        } else {
            no_improvement_count++;
        }
    }

    // Local search for each solution
    if (is_local_search) {
        torch::Tensor batch_idx = torch::arange(batch_size).to(device);
        while (true) {
            torch::Tensor selected_solutions = best_solutions.index({batch_idx});
            torch::Tensor selected_full_energy_2D = full_energy_2D.index({batch_idx});
            torch::Tensor selected_delta_energy_sign = torch::logical_not(selected_solutions) * 2 - 1;
            torch::Tensor selected_delta_energies =
                selected_delta_energy_sign *
                torch::sum(selected_solutions.unsqueeze(2) * selected_solutions.unsqueeze(1) * selected_full_energy_2D,
                           {2});
            torch::Tensor selected_accept_mask = selected_delta_energies < 0;
            torch::Tensor continue_iter_mask_local = selected_accept_mask.any({1});
            if (!continue_iter_mask_local.any().item<bool>()) {
                break;
            }

            std::vector<int64_t> new_batch_idx_vector;
            std::vector<int64_t> batch_idx_inner_vector;
            for (int b = 0; b < batch_idx.numel(); b++) {
                if (continue_iter_mask_local[b].item<bool>()) {
                    new_batch_idx_vector.push_back(batch_idx[b].item<int64_t>());
                    batch_idx_inner_vector.push_back(b);
                }
            }
            batch_idx = torch::tensor(new_batch_idx_vector, torch::kInt64).to(device);
            torch::Tensor batch_idx_inner = torch::tensor(batch_idx_inner_vector, torch::kInt64).to(device);

            selected_delta_energies = selected_delta_energies.index({batch_idx_inner});
            torch::Tensor min_delta_energy_indices = torch::argmin(selected_delta_energies, 1);
            best_solutions.index_put_(
                {batch_idx, min_delta_energy_indices},
                torch::logical_not(selected_solutions.index({batch_idx_inner, min_delta_energy_indices})));
        }
    }

    if (flag_evolution_history && !history_file_path.empty()) {
        torch::save(ha_history_tensor.to(torch::kCPU), history_file_path);
    }

    return best_solutions;
}

namespace py = pybind11;
PYBIND11_MODULE(simulated_annealing_qubo, m) {
    m.def("simulated_annealing_qubo", &simulated_annealing_qubo, "Simulated Annealing for QUBO", py::arg("Q"),
          py::arg("initial_temperature") = 1000.0, py::arg("alpha") = 0.99, py::arg("cutoff_temperature") = 0.001,
          py::arg("iterations_per_t") = 10, py::arg("patience") = 10, py::arg("is_local_search") = true,
          py::arg("flag_evolution_history") = false, py::arg("history_file_path") = "",
          py::arg("rand_seed") = c10::nullopt, py::arg("is_input_check") = true, py::arg("csa_init_x") = c10::nullopt);
}
