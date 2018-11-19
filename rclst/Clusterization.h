#pragma once

#include <string>
#include <vector>
#include <dlib/clustering.h>
#include <dlib/svm_threaded.h>

struct Clusterization {
    static const int                N = 7;
    static const std::array<int, N> precision;

    using sample_type = dlib::matrix<double, N, 1>;
    using ovo_trainer = dlib::one_vs_one_trainer<dlib::any_trainer<sample_type>>;
    using poly_kernel = dlib::polynomial_kernel<sample_type>;
    using rbf_kernel  = dlib::radial_basis_kernel<sample_type>;

    Clusterization(size_t clusters_num, std::string model_file_name);

    void execute();

private:
    void init();
    void clusterize();
    void save();

    void parse_data();
    void normalize_data();
    void fill_samples();
    void save_raw_data();
    void save_cf();
    void save_df();

    std::string m_model_file_name;
    size_t      m_clusters_num;

    std::array<std::vector<double>, N>   m_raw_data{};
    std::array<std::vector<double>, N>   m_norm_data{};
    std::array<std::array<double, 2>, N> m_norm_cf{};

    std::vector<sample_type> m_samples{};
    std::vector<double>      m_labels{};
};

