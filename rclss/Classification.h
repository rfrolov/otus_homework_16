#pragma once

#include <string>
#include <dlib/clustering.h>
#include <dlib/svm_threaded.h>

struct Classification {
    static const int N = 7;
    static const std::array<int, N> precision;

    using sample_type = dlib::matrix<double, N, 1>;
    using ovo_trainer = dlib::one_vs_one_trainer<dlib::any_trainer<sample_type>>;
    using poly_kernel = dlib::polynomial_kernel<sample_type>;
    using rbf_kernel  = dlib::radial_basis_kernel<sample_type>;

    explicit Classification(std::string model_file_name);

    void process_data(const std::string &line);
private:

    void read_cf();
    double get_label(const sample_type &sample) const;
    void get_data(size_t num, const sample_type &sample);
    void sort_data(std::vector<std::array<double, N>> &raw_data, const sample_type &sample) const;
    void out_data(const std::vector<std::array<double, N>> &raw_data) const;

    std::string                          m_model_file_name;
    std::array<std::array<double, 2>, N> norm_cf;

    dlib::one_vs_one_decision_function<ovo_trainer, dlib::decision_function<poly_kernel>,
            dlib::decision_function<rbf_kernel>> df;
};
