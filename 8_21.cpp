/*
Equation 8.14 gives T = n^3/p + 10*log2(p) + 2*n^2/sqrt(p).
For each p, increase integer n until the next value would take over 512
time units. The last row also requires n to be divisible by sqrt(p).

p                  1    4   16   64   256   1024   4096
Largest n          7   11   17   26    39     58     86
Equal-block n      7   10   16   24    32     32     64

At p = 1, the printed formula still charges for communication. Actual
serial multiplication takes n^3, so an 8 by 8 matrix fits exactly.
An unlimited supply of processes cannot give unlimited n in this model:
10*log2(p) <= 512 bounds p, and n^3 <= 512*p then bounds n.
*/

#include <cmath>
#include <iomanip>
#include <iostream>

double tm(int n, int p) {
    double x = n;
    return x * x * x / p + 10 * std::log2(p) + 2 * x * x / std::sqrt(p);
}

int main() {
    std::cout << "p  max_n  time  equal_block_n  equal_block_time\n";
    for (int p : {1, 4, 16, 64, 256, 1024, 4096}) {
        int n = 0;
        while (tm(n + 1, p) <= 512) ++n;
        int q = static_cast<int>(std::sqrt(p));
        int m = n - n % q;
        std::cout << p << ' ' << n << ' ' << std::fixed << std::setprecision(3)
                  << tm(n, p) << ' ' << m << ' ' << tm(m, p) << '\n';
    }
}
