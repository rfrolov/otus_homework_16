#include <iostream>
#include "Classification.h"

int main(int argc, char **argv) {

    if (argc != 2) {
        std::cerr << "Неверное число параметров.\nФормат вызова: rclss modelfname" << std::endl;
        return 1;
    }

    Classification app(argv[1]);

    for (std::string line; std::getline(std::cin, line);) {
        app.process_data(line);
    }

    return 0;
}



