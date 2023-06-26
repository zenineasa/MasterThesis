#!/bin/sh

# Run the following command for clean build:
# ./script.sh 100

command() {
    # Change the directory to the location of this script file
    cd "$(dirname "$0")"

    # Copy over file content to Ipopt folder
    # For simplicity, let's just replace the existing ScalableProblems directory
    rm -rf ../Ipopt/examples/ScalableProblems
    mkdir ../Ipopt/examples/ScalableProblems
    cp * ../Ipopt/examples/ScalableProblems

    # Go and build
    cd ../Ipopt/build

    if [ $1 -eq 100 ]
    then
            echo "clean build..."
            ./sh.sh
            make clean
            make
            make install
    else
            echo "No clean build..."
    fi

    cd examples/ScalableProblems
    make

    echo ""
    echo "Build should be complete by now. To check this, you could run the following commands:"
    echo "cd $(pwd)"
    echo './solve_problem MBndryCntrlDiri 100 0.01 -1e20 3.5 0. 10. -20. "3.+5.*(x1*(x1-1.)*x2*(x2-1.))"'
    echo './solve_problem MBndryCntrlNeum 100 0.01 -1e20 2.071 3.7 4.5 4.1 "2.-2.*(x1*(x1-1.)+x2*(x2-1.))"'
    echo ""
}

command $1
