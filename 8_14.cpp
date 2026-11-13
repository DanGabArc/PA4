/*
Use the book's pivot rule: choose the largest absolute value in the
current row and swap columns. Assume p divides n and q = sqrt(p) is an
integer. Count one unit per division, multiplication, subtraction or
comparison. Local moves are free. The mesh uses synchronous phases,
full-duplex links and all ports. A one-hop message costs L(w) = ts + tw*w.

For step k = 0,...,n-2, let m = n-1-k. Write o(j) for the owner of index j
and H(t) for the largest number of the last t indices on any process.
Block:  b = n/p, o(j) = floor(j/b), H(t) = min(t,b).
Cyclic: o(j) = j mod p, H(t) = ceil(t/p).

Let h = H(m). If o(k) is at (i,j), a broadcast across its row and then
along the columns takes d hops, d = max(i,q-1-i) + max(j,q-1-j).
The busiest process updates m*h entries, so its update takes 2*m*h.
Sum each expression below over all k to get the total time.

Layout    Without pivot                 With partial pivot
Rows      m + d*L(m) + 2*m*h           2*m + d*L(m+1) + 2*m*h
Columns   h + d*L(m+1) + 2*m*h         h + d*L(m+1) + 2*m*h
                                       + H(m+1)-1 + G + s*L(n)

(a) row/block: use the row formula and block o,H.
(b) row/cyclic: use the row formula and cyclic o,H.
(c) column/block: use the column formula and block o,H.
(d) column/cyclic: use the column formula and cyclic o,H.

With rows, one owner divides m entries. Pivoting adds m comparisons and
one index to its broadcast. With columns, the owner broadcasts the pivot
and m values below it. Each rank divides up to h entries. To find a pivot,
each rank also makes up to H(m+1)-1 local comparisons. Reduce a two-word
(value,index) pair to a corner, then broadcast the winning index:
G = 2*(q-1)*(L(2) + 1 + L(1)).
Here s is the Manhattan distance between the two column owners. Exchanging
their full columns costs s*L(n), so this time depends on the chosen pivots.
*/

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    int n = 8, p = 4;
    double ts = 1, tw = 1;
    bool read = argc == 2 && std::string(argv[1]) == "--stdin";
    if ((argc != 1 && !read) || (read && !(std::cin >> n >> p >> ts >> tw))) {
        std::cerr << "Use --stdin with n, p, ts, tw and n-1 zero-based pivot columns.\n";
        return 1;
    }
    if (n <= 0 || p <= 0 || n % p != 0 || ts < 0 || tw < 0
        || !std::isfinite(ts) || !std::isfinite(tw)) {
        std::cerr << "Use positive n and p, p dividing n, and nonnegative times.\n";
        return 1;
    }
    int q = static_cast<int>(std::sqrt(p));
    if (q * q != p) {
        std::cerr << "The number of processes must be a square.\n";
        return 1;
    }
    std::vector<int> piv(n - 1, n - 1);
    if (read) {
        for (int k = 0; k < n - 1; ++k) {
            if (!(std::cin >> piv[k]) || piv[k] < k || piv[k] >= n) {
                std::cerr << "Pivot column at step k must be between k and n-1.\n";
                return 1;
            }
        }
    }
    auto msg = [=](int w) { return ts + tw * w; };
    double g = 2.0 * (q - 1) * (msg(2) + 1 + msg(1));
    int b = n / p;
    std::cout << std::setprecision(12);
    std::cout << "Layout                    Without pivot   With pivot\n";
    for (bool col : {false, true}) {
        for (bool cyc : {false, true}) {
            auto own = [=](int j) { return cyc ? j % p : j / b; };
            auto load = [=](int m) {
                return cyc ? m / p + (m % p != 0) : std::min(b, m);
            };
            double t0 = 0, t1 = 0;
            for (int k = 0; k < n - 1; ++k) {
                int m = n - 1 - k, h = load(m);
                int u = own(k), v = own(piv[k]);
                int i = u / q, j = u % q;
                int d = std::max(i, q - 1 - i) + std::max(j, q - 1 - j);
                double upd = 2.0 * m * h;
                if (!col) {
                    t0 += m + d * msg(m) + upd;
                    t1 += 2.0 * m + d * msg(m + 1) + upd;
                } else {
                    int s = std::abs(i - v / q) + std::abs(j - v % q);
                    double base = h + d * msg(m + 1) + upd;
                    t0 += base;
                    t1 += base + load(m + 1) - 1 + g + s * msg(n);
                }
            }
            std::string name = std::string(col ? "column " : "row ") + (cyc ? "cyclic" : "block");
            std::cout << std::left << std::setw(24) << name << std::right
                      << std::setw(15) << t0 << std::setw(13) << t1 << '\n';
        }
    }
}
