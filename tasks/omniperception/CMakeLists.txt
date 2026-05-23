cmake_minimum_required(VERSION 3.16)

# omniperception 自身不调用 OpenVINO，仅链 io/tools 依赖
add_library(omniperception OBJECT
    decider.cpp
    perceptron.cpp
)
