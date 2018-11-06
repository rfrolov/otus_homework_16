#include <utility>
#include <iostream>
#include <vector>
#include "Classification.h"
#include <boost/algorithm/string.hpp>

const std::array<int, Classification::N> Classification::precision{{6, 6, 0, 2, 2, 2, 0}};

Classification::Classification(std::string model_file_name) :
        m_model_file_name{std::move(model_file_name)} {
    read_cf();
    dlib::deserialize(m_model_file_name) >> df;
}

void Classification::process_data(const std::string &line) {
    std::vector<std::string> data;
    boost::split(data, line, boost::is_any_of(";"));

    sample_type sample;

    for (int i = 0; i < N; ++i) {
        if (!data[i].empty()) {
            sample(i) = std::stod(data[i]);
        } else {
            sample(i) = 0;
        }
    }

    auto label = get_label(sample);
    get_data(label, sample);
}

void Classification::read_cf() {
    std::ifstream fs(m_model_file_name + ".cf");

    int n;
    fs >> n;
    for (int i = 0; i < N; ++i) {
        fs >> norm_cf[i][0];
        fs >> norm_cf[i][1];
    }
}
double Classification::get_label(const Classification::sample_type &sample) const {
    auto test_sample = sample;

    for (int i = 0; i < N; ++i) {
        test_sample(i) = test_sample(i) / norm_cf[i][1];
    }

    return df(test_sample);
}

void Classification::get_data(size_t num, const Classification::sample_type &sample) {
    std::vector<std::array<double, N>> raw_data;
    std::ifstream                      fs(m_model_file_name + "." + std::to_string(num));

    for (std::string line; std::getline(fs, line);) {

        std::vector<std::string> data;
        boost::split(data, line, boost::is_any_of(";"));

        std::array<double, N> smpl;

        std::generate_n(smpl.begin(), N, [i = 0, &data]() mutable { return std::stod(data[i++]); });

        raw_data.emplace_back(smpl);
    }

    sort_data(raw_data, sample);
    out_data(raw_data);
}

void Classification::sort_data(std::vector<std::array<double, N>> &raw_data, const sample_type &sample) const {
    auto x = sample(0);
    auto y = sample(1);

    std::sort(raw_data.begin(), raw_data.end(), [x, y](const auto &a, const auto &b) {
        return (std::sqrt(std::pow(x - a[0], 2) + std::pow(y - a[1], 2)) <
                std::sqrt(std::pow(x - b[0], 2) + std::pow(y - b[1], 2)));
    });
}

void Classification::out_data(const std::vector<std::array<double, N>> &raw_data) const {
    for (const auto &data : raw_data) {
        std::cout << std::fixed << std::setprecision(precision[0]);
        std::cout << data[0] << ";" << data[1];
        for (auto j = 2; j < N; ++j) {
            std::cout << std::fixed << std::setprecision(precision[j]);
            std::cout << ";" << data[j];
        }
        std::cout << "\n";
    }
}