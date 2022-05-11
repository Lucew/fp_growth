![Banner](images/banner.jpg)
# FP Growth
This project is a simple and pure Python implementation of the FP-Grwoth algorithm used for knowledge discovery in 
databases. I implemented this as a learning example for recursive tree algorithms.

**The main code only uses built-in python libraries and has no dependencies**.

You can find the used example [here](https://www.mygreatlearning.com/blog/understanding-fp-growth-algorithm/).

Cheers, Lucas.

# Usage
Just run the [file](FpGrowth.py) with:

`python FpGrowth.py`

to see an example from the previously mentioned website.

The main function to use is [fp_growth](./FpGrowth.py#:~:text=def%20fp_growth).

In order to runt the [tests](test/Test_FpGrowth.py), install the [requirements](test/requirements.txt). using pip (or any other package management tool):

`pip -r install test\requirements.txt`


# Testing
One can compare the package given in this repo with the 
[mlxtend implementation](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/). In order to
run the prebuild tests one can install the requirements in the test repository with:

`pip -r install test\requirements.txt`

and then run:

`python test\Test_FpGrwoth.py`

You can also introduce your own tests by using the `test_own_algorithm` function in the source of the
[test file](test/Test_FpGrowth.py#:~:text=def%20test_own_algorithm).

