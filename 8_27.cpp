/*
Let p = q^2, b = n/q, and assume q divides n. Each process owns a b by b
output block. For C[i,j], k' = (i+j+k) mod n visits every inner index once.
For each input matrix, b of the n reads per output are local. A process
does b^2*n updates and makes D = 2*b^2*(n-b) remote reads.

Let L be the local access cost, tw the extra remote cost, and c the cost
of a multiply-add. Counting every access at its local price first gives
F = b^2*(L + n*(c + 4*L)). The access-count estimate is T = F + D*tw.

If ranks wait after each input read, work on the same offset in each
output block. The staggered indices send reads to different banks. For
q > 1 at least one read in each phase is remote, so the synchronized time
is F + 2*b^2*n*tw. For q = 1 it is F. Both give Theta(n^3/p) for fixed costs.
*/

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

int main(int argc, char **argv) {
    int n = 8, p = 4;
    double l = 1, w = 5, c = 1;
    if (argc != 1) {
        std::stringstream in;
        for (int i = 1; i < argc; ++i) in << argv[i] << ' ';
        if (argc != 6 || !(in >> n >> p >> l >> w >> c) || !(in >> std::ws).eof() ||
            n < 1 || p < 1 || p > 1LL * n * n || l < 0 || w < 0 || c < 0 ||
            !std::isfinite(l + w + c)) {
            std::cerr << "Usage: 8_27 n p tl tw tc\n";
            return 1;
        }
    }
    int q = static_cast<int>(std::sqrt(p));
    long long b = (n + 1LL * q - 1) / q;
    long long m = b * q;
    long double z = 1.0L * b * b;
    long double d = 2 * z * (m - b);
    long double t = z * (l + m * (c + 4.0L * l)) + d * w;
    long double s = z * (l + m * (c + 4.0L * l + (q == 1 ? 0 : 2.0L * w)));
    std::cout << std::setprecision(12)
              << "Active processors = " << q * q << '\n'
              << "Matrix order after padding = " << m << '\n'
              << "Remote reads per processor = " << d << '\n'
              << "Time from the access counts = " << t << '\n'
              << "Time with synchronized read phases = " << s << '\n';
}
