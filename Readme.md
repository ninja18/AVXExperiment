## Experiment to understand AVX
### Setup
To calculate the complex dot product with two vectors having million complex component. More detailed setup in the reference.
Tested in intel i7 1165G7 processor with AVX512 and FMA compiled with GCC 12.2.0 in Debian.

### Observations
* Since this is a simple program with no conditionals, the compiler was able to vectorize almost completely in Release mode. The performance difference between naive and vectorized code is just 31% with both AVX2 and AVX512.
* Performance difference in Debug mode where the compiler did not vectorize was upto 400% with AVX512 and 250% with AVX2.
* With `-O3` enabled the compiler was even able to do Fuse multiply add which was not present in lower optimization levels.
* In manual vectorized code, only the result value made a difference with FMA and no performance gain at all. The difference in result was `0.30` between naive and AVX without FMA and `0.10` with FMA.

### Reference
* [Basic SIMD and AVX tutorial](https://youtu.be/AT5nuQQO96o?si=HZPEoKausZDhGOou)
