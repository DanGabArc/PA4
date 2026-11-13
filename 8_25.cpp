/*
Let L be the local access time, R = L + tw the remote access time,
and c the time for one multiply-add. Assume each memory bank handles
one read at a time. At step k, all n processes in row i read A[i,k].
One read is local and n-1 are remote, so this phase takes L+(n-1)*R.
Reading B[k,j] has the same cost. Each update also reads and writes C
locally. Counting its initial write gives

T = L + n*(c + 2*L + 2*(L + (n-1)*R)) = Theta(n^2).

The bank rule matters. If a bank serves all simultaneous reads at once,
each step takes constant time and the total is Theta(n).
*/

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

int main(int argc, char **argv) {
    int n = 8;
    double l = 1, w = 5, c = 1;
    if (argc != 1) {
        std::stringstream in;
        for (int i = 1; i < argc; ++i) in << argv[i] << ' ';
        if (argc != 5 || !(in >> n >> l >> w >> c) || !(in >> std::ws).eof() ||
            n < 1 || l < 0 || w < 0 || c < 0 ||
            !std::isfinite(l + w + c)) {
            std::cerr << "Usage: 8_25 n tl tw tc\n";
            return 1;
        }
    }
    long double r = static_cast<long double>(l) + w;
    long double t = l + n * (c + 4.0L * l + 2.0L * (n - 1) * r);
    std::cout << std::setprecision(12)
              << "Processors = " << 1LL * n * n << '\n'
              << "Remote requests per input word in its read phase = " << n - 1 << '\n'
              << "Time with one reader per bank = " << t << '\n';
}
