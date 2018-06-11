#! /bin/bash
g++ VirtualScanner.cpp main.cpp -DCGAL_NO_PRECONDITIONS -std=c++11 -I/usr/include/eigen3 -o virtual-scanner -lboost_system -lCGAL -fopenmp
