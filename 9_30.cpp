/*
Let m=n/p, with m>=p and p a power of two. Sort each local block and
choose samples at one-based positions ceil(j*m/p), j=1,...,p-1. Attach
the original index to every key so equal values have a fixed order.

Pad each sample block with one high sentinel. Run bitonic merges on
groups of g=2,4,...,p ranks. View a group as g rows of p keys. The first
layers compare keys in the same column. Transpose, merge each bitonic
column, transpose back, then merge each bitonic row. The final list has
all p*(p-1) real samples first. Choose global sample ranks j*(p-1),
counting from one, for j=1,...,p-1. Share these splitters with all ranks.

One transpose costs g message starts and moves p keys per rank. The local
merges cost O(p) per stage. With concurrent full-bandwidth exchanges and
fixed message costs, all sample stages together take Theta(p log p).

Use binary search to split each local block, send the parts to their
bucket owners, and sort each received bucket. With balanced traffic,
the exchange costs O(m+p) and each final bucket has O(m) keys. Hence
T = Theta(m*log2(m) + p*log2(p)),
S = Theta(n*log2(n) / (m*log2(m) + p*log2(p))), and E = S/p.
These bounds assume the input starts distributed. Skewed traffic can
increase the exchange time.
*/

#include <mpi.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <climits>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

using Key = std::array<int, 2>;
static_assert(sizeof(Key) == 2 * sizeof(int));

void bitmerge(std::vector<Key>& a, std::vector<Key>& tmp, bool up) {
    std::rotate(a.begin(), std::min_element(a.begin(), a.end()), a.end());
    auto mid = std::max_element(a.begin(), a.end()) + 1;
    std::reverse(mid, a.end());
    std::merge(a.begin(), mid, mid, a.end(), tmp.begin());
    if (!up) std::reverse(tmp.begin(), tmp.end());
    a.swap(tmp);
}

void transpose(const std::vector<Key>& a, std::vector<Key>& t, int id, int g) {
    int z = static_cast<int>(a.size()) / g;
    int lane = id % g;
    for (int k = 0; k < g; ++k) {
        int dst = lane ^ k;
        int peer = id - lane + dst;
        MPI_Sendrecv(a.data() + dst * z, z, MPI_2INT, peer, 0,
                     t.data() + dst * z, z, MPI_2INT, peer, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int id, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int n = 0, ok = 1;
    std::vector<int> a;
    if (id == 0) {
        long long z = argc > 1 ? argc - 1 : std::max(16LL, 1LL * p * p);
        ok = !(p & (p - 1)) && z <= INT_MAX && z % p == 0 && z >= 1LL * p * p;
        if (ok) {
            n = static_cast<int>(z);
            a.resize(n);
            for (int i = 0; i < n && ok; ++i) {
                if (argc == 1) {
                    a[i] = static_cast<int>((17LL * i + 8) % 29) - 14;
                } else {
                    const char* s = argv[i + 1];
                    const char* e = s + std::strlen(s);
                    auto v = std::from_chars(s, e, a[i]);
                    ok = v.ec == std::errc{} && v.ptr == e;
                }
            }
        }
        if (!ok) std::cerr << "Use a power of two ranks and integer keys. "
                            << "n must be divisible by p and at least p*p.\n";
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok) {
        MPI_Finalize();
        return 1;
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int m = n / p;
    std::vector<int> x(m);
    MPI_Scatter(a.data(), m, MPI_INT, x.data(), m, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<Key> b(m);
    for (int i = 0; i < m; ++i) b[i] = {x[i], id * m + i};
    std::sort(b.begin(), b.end());

    std::vector<Key> s(p), t(p), buf(p), piv(p);
    for (int j = 1; j < p; ++j) s[j - 1] = b[(1LL * j * m - 1) / p];
    s.back() = {INT_MAX, INT_MAX};
    if (id % 2 == 1) std::reverse(s.begin(), s.end());
    for (int g = 2; g <= p; g *= 2) {
        bool up = (id & g) == 0;
        int z = p / g;
        transpose(s, t, id, g);
        std::vector<Key> col(g), tmp(g);
        for (int j = 0; j < z; ++j) {
            for (int i = 0; i < g; ++i) col[i] = t[i * z + j];
            bitmerge(col, tmp, up);
            for (int i = 0; i < g; ++i) t[i * z + j] = col[i];
        }
        transpose(t, s, id, g);
        bitmerge(s, buf, up);
    }
    if (p > 1) {
        Key v = id < p - 1 ? s[p - id - 2] : s.back();
        MPI_Allgather(&v, 1, MPI_2INT, piv.data(), 1, MPI_2INT, MPI_COMM_WORLD);
    }

    std::vector<int> sc(p), sd(p), rc(p), rd(p);
    int pos = 0;
    for (int i = 0; i < p; ++i) {
        sd[i] = pos;
        if (i == p - 1) pos = m;
        else pos = static_cast<int>(std::upper_bound(b.begin() + pos, b.end(), piv[i]) - b.begin());
        sc[i] = pos - sd[i];
    }
    MPI_Alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::exclusive_scan(rc.begin(), rc.end(), rd.begin(), 0);
    int cnt = rd.back() + rc.back();
    std::vector<Key> y(cnt);
    MPI_Alltoallv(b.data(), sc.data(), sd.data(), MPI_2INT, y.data(), rc.data(),
                  rd.data(), MPI_2INT, MPI_COMM_WORLD);
    std::sort(y.begin(), y.end());

    std::vector<int> val(cnt), sz(p), off(p);
    for (int i = 0; i < cnt; ++i) val[i] = y[i][0];
    MPI_Gather(&cnt, 1, MPI_INT, sz.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (id == 0) std::exclusive_scan(sz.begin(), sz.end(), off.begin(), 0);
    MPI_Gatherv(val.data(), cnt, MPI_INT, a.data(), sz.data(), off.data(), MPI_INT, 0, MPI_COMM_WORLD);
    if (id == 0) {
        std::cout << "Splitters (key, original index):";
        for (int i = 0; i < p - 1; ++i) std::cout << " (" << piv[i][0] << ", " << piv[i][1] << ')';
        std::cout << "\nSorted:";
        for (int v : a) std::cout << ' ' << v;
        std::cout << '\n';
    }
    MPI_Finalize();
}
