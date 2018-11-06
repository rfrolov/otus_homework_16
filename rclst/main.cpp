#include <iostream>
#include <algorithm>
#include "Clusterization.h"

int main(int argc, char **argv) {

    if (argc != 3) {
        std::cerr << "Неверное число параметров.\nФормат вызова: rclst <n> modelfname" << std::endl;
        return 1;
    }

    size_t      clusters_num{0};
    std::string str_clusters_num{argv[1]};

    if (std::all_of(str_clusters_num.begin(), str_clusters_num.end(), ::isdigit)) {
        clusters_num = (size_t) strtoll(str_clusters_num.c_str(), nullptr, 0);
    }

    if (clusters_num == 0) {
        std::cerr << "Неверный параметр <n>" << std::endl;
        return 1;
    }

    std::string model_file_name = argv[2];

    Clusterization app(clusters_num, model_file_name);
    app.execute();

    return 0;
}



