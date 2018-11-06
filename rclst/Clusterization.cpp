#include <utility>
#include <iostream>
#include "Clusterization.h"
#include <boost/algorithm/string.hpp>


const std::array<int, Clusterization::N> Clusterization::precision{{6, 6, 0, 2, 2, 2, 0}};

Clusterization::Clusterization(size_t clusters_num, std::string model_file_name) :
        m_clusters_num{clusters_num}
        , m_model_file_name{std::move(model_file_name)} {
}

void Clusterization::execute() {
    init();
    clusterize();
    save();
}

void Clusterization::parse_data() {
    std::vector<std::string> data{};

    for (std::string line; std::getline(std::cin, line);) {
        data.clear();

        boost::split(data, line, boost::is_any_of(";"));
        if (data.size() != 8) { continue; }

        for (int i = 0; i < N; ++i) {
            m_raw_data[i].emplace_back(strtod(data[i].c_str(), nullptr));
        }

        auto val = strtod(data[N].c_str(), nullptr);
        if (m_raw_data[N - 1].back() == 1 || m_raw_data[N - 1].back() == val) {
            m_raw_data[N - 1].back() = 0;
        } else {
            m_raw_data[N - 1].back() = 1;
        }
    }
}

void Clusterization::normalize_data() {
    std::array<double, 2> dists{};

    for (int i = 0; i < 2; ++i) {
        m_norm_data[i].reserve(m_raw_data[i].size());

        auto min_max = std::minmax_element(m_raw_data[i].begin(), m_raw_data[i].end());

        dists[i]      = *min_max.second - *min_max.first;
        m_norm_cf[i][0] = *min_max.first;
    }

    auto dist = std::max(dists[0], dists[1]);

    for (int i = 0; i < 2; ++i) {
        std::transform(m_raw_data[i].begin(), m_raw_data[i].end(), back_inserter(m_norm_data[i]),
                       [dist](double val) { return val / dist; });
        m_norm_cf[i][1] = dist;
    }


    for (int i = 2; i < N; ++i) {
        m_norm_data[i].reserve(m_raw_data[i].size());

        auto min_max = std::minmax_element(m_raw_data[i].begin(), m_raw_data[i].end());

        auto min = *min_max.first;
        auto max = *min_max.second;

        std::transform(m_raw_data[i].begin(), m_raw_data[i].end(), back_inserter(m_norm_data[i]),
                       [max](double val) { return val / max; });
        m_norm_cf[i][0] = min;
        m_norm_cf[i][1] = max;
    }
}

void Clusterization::fill_samples() {
    for (size_t i = 0; i < m_norm_data[0].size(); ++i) {
        sample_type m;

        for (size_t j = 0; j < N; ++j) {
            m(j) = m_norm_data[j][i];
        }

        m_samples.push_back(std::move(m));
    }
}

void Clusterization::clusterize() {
    using kernel_type = dlib::radial_basis_kernel<sample_type>;

    dlib::kcentroid<kernel_type> kc(kernel_type(0.1), 0.01, 8);
    dlib::kkmeans<kernel_type>   test(kc);

    std::vector<sample_type> initial_centers;

    test.set_number_of_centers(m_clusters_num);
    pick_initial_centers(m_clusters_num, initial_centers, m_samples, test.get_kernel());

    test.train(m_samples, initial_centers);

    for (auto const &s: m_samples) {
        m_labels.emplace_back(test(s));
    }
}

void Clusterization::init() {
    parse_data();
    normalize_data();
    fill_samples();
}

void Clusterization::save() {
    save_raw_data();
    save_cf();
    save_df();
}

void Clusterization::save_raw_data() {
    std::vector<std::ofstream> fs;

    for (int i = 0; i < m_clusters_num; ++i) {
        std::string fn = m_model_file_name + "." + std::to_string(i);
        fs.emplace_back(std::ofstream{fn});
    }

    for (size_t i = 0; i < m_samples.size(); ++i) {
        auto index = m_labels[i];
        fs[index] << std::fixed << std::setprecision(precision[0]);
        fs[index] << m_raw_data[0][i] << ";" << m_raw_data[1][i];
        for (auto j = 2; j < N; ++j) {
            fs[index] << std::fixed << std::setprecision(precision[j]);
            fs[index] << ";" << m_raw_data[j][i];
        }
        fs[index] << "\n";
    }
}

void Clusterization::save_cf() {
    std::ofstream fs(m_model_file_name + ".cf");

    fs << N << "\n";
    for (int i = 0; i < N; ++i) {
        fs << m_norm_cf[i][0] << " " << m_norm_cf[i][1] << "\n";
    }
}

void Clusterization::save_df() {
    ovo_trainer trainer;

    dlib::krr_trainer<rbf_kernel>     rbf_trainer;
    dlib::svm_nu_trainer<poly_kernel> poly_trainer;

    poly_trainer.set_kernel(poly_kernel(0.1, 1, 2));
    rbf_trainer.set_kernel(rbf_kernel(0.1));

    trainer.set_trainer(rbf_trainer);
    trainer.set_trainer(poly_trainer, 1, 2);

    dlib::one_vs_one_decision_function<ovo_trainer> df = trainer.train(m_samples, m_labels);

    dlib::one_vs_one_decision_function<ovo_trainer, dlib::decision_function<poly_kernel>,
            dlib::decision_function<rbf_kernel>> df2;

    df2 = df;
    dlib::serialize(m_model_file_name) << df2;
}