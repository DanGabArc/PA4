/*
Each process owns n/p columns of A and the matching part of x. It computes
a partial sum for every output row. A reduce-scatter adds partial sums for
the same row and leaves each process with n/p entries of y.

Assume p divides n and p is a power of two. Each process does n^2/p
multiplications. The reduction sends n/2 + n/4 + ... + n/p = n(1-1/p)
words in log2(p) rounds. With rows, the allgather of x sends the same
sizes in reverse order. Both layouts have the same time in the book's model:
T = n^2/p + ts*log2(p) + tw*n*(1-1/p).
*/

#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int r, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int n = 4, ok = 1;
    std::vector<double> a, x;
    if (r == 0) {
        if (argc == 1) {
            a = {1, 2, 0, 3, -1, 1, 4, 0, 2, 0, 1, -1, 0, 3, 2, 2};
            x = {2, -1, 3, 1};
        } else if (argc == 2 && std::string(argv[1]) == "--stdin"
                   && std::cin >> n && n > 0 && n <= 46340) {
            a.resize(n * n);
            x.resize(n);
            for (double &v : a) ok &= bool(std::cin >> v) && std::isfinite(v);
            for (double &v : x) ok &= bool(std::cin >> v) && std::isfinite(v);
        } else {
            ok = 0;
        }
        if (!ok) std::cerr << "Use --stdin with n, an n by n matrix, and n vector entries.\n";
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok) {
        MPI_Finalize();
        return 1;
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int> cnt(p), pos(p), len(p), off(p);
    for (int k = 0; k < p; ++k) {
        cnt[k] = n / p + (k < n % p);
        pos[k] = k * (n / p) + std::min(k, n % p);
        len[k] = n * cnt[k];
        off[k] = n * pos[k];
    }
    std::vector<double> buf;
    if (r == 0) {
        buf.resize(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) buf[j * n + i] = a[i * n + j];
    }
    std::vector<double> b(len[r]), u(cnt[r]), v(n, 0), y(cnt[r]);
    MPI_Scatterv(buf.data(), len.data(), off.data(), MPI_DOUBLE, b.data(), len[r],
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(x.data(), cnt.data(), pos.data(), MPI_DOUBLE, u.data(), cnt[r],
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (cnt[r] > 0) {
        for (int i = 0; i < n; ++i) v[i] = b[i] * u[0];
        for (int j = 1; j < cnt[r]; ++j)
            for (int i = 0; i < n; ++i) v[i] += b[j * n + i] * u[j];
    }
    MPI_Reduce_scatter(v.data(), y.data(), cnt.data(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    std::vector<double> out(r == 0 ? n : 0);
    MPI_Gatherv(y.data(), cnt[r], MPI_DOUBLE, out.data(), cnt.data(), pos.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (r == 0) {
        std::cout.precision(std::numeric_limits<double>::max_digits10);
        for (int i = 0; i < n; ++i) std::cout << (i ? " " : "") << out[i];
        std::cout << '\n';
    }
    MPI_Finalize();
}
