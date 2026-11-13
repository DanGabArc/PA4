/*
Let q = sqrt(p), b = n/q. A process needs q blocks from its row of A and
q from its column of B. It owns one of each, so it copies H = 2*(q-1)*b^2
words. Cycle the source blocks so that each bank serves one reader per
phase. Each copy needs a remote read (L+tw) and a local write (L).
After that, all multiply-adds use local memory.

F = b^2*(L + n*(c + 4*L))
T28 = F + H*(2*L + tw)

The direct method in 8.27 reads b*H remote words. Its access-count time
is F + b*H*tw. Copying saves H*((b-1)*tw - 2*L) time units when this
quantity is positive, and needs H extra local words per process.
For n=8, p=4, L=c=1 and tw=5, the times are 1296 (direct) and 880 (copy).
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
            std::cerr << "Usage: 8_28 n p tl tw tc\n";
            return 1;
        }
    }
    int q = static_cast<int>(std::sqrt(p));
    long long b = (n + 1LL * q - 1) / q;
    long long m = b * q;
    long double z = 1.0L * b * b;
    long double h = 2 * z * (q - 1);
    long double t = z * (l + m * (c + 4.0L * l));
    std::cout << std::setprecision(12)
              << "Active processors = " << q * q << '\n'
              << "Matrix order after padding = " << m << '\n'
              << "Remote words copied per processor = " << h << '\n'
              << "Time with prefetch = " << t + h * (2.0L * l + w) << '\n'
              << "Direct time from the access counts = " << t + h * b * w << '\n';
}
