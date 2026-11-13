/*
Number the ring clockwise and place wire r on rank r. For p = 2^d ranks,
partners at stride 2^j are 2^j hops apart. Send the lower half clockwise
and the upper half counterclockwise on a bidirectional ring. Stride 2^j
occurs d-j times, so the number of one-hop rounds is
sum(j=0..d-1) (d-j)*2^j = 2*p-d-2.

For p=n this gives T = Theta(n), speedup Theta(log n), and efficiency
Theta(log n/n). Some input must send a key across half the ring, so the
mapping has the best possible order of communication time.

For p<n let m=n/p. Each rank sorts m keys, then merges and keeps half
at each comparison. Sending m keys per hop takes Theta(m*p) in total.
The local merges take O(m*d^2), which is also O(m*p). Thus
T = Theta(m*log2(m+1) + n), S = Theta(n*log2(n)/T), E = S/p.

Compared with serial work Theta(n log n), the cost p*T stays optimal
for p = O(log n). The largest useful order is therefore Theta(log n).
At fixed efficiency log2(n) is proportional to p. Writing n = 2^(a*p)
gives isoefficiency W = Theta(p*2^(a*p)) for a fixed constant a > 0.
*/

#include <algorithm>
#include <charconv>
#include <iostream>
#include <mpi.h>
#include <string_view>
#include <vector>

using namespace std;

void exchange(const vector<int>& a, vector<int>& b, int id, int p, int h) {
    int left = (id + p - 1) % p;
    int right = (id + 1) % p;
    int m = static_cast<int>(a.size());
    vector<int> cw = a, ccw = a, u(m), v(m);
    for (int t = 0; t < h; ++t) {
        int x = (id - t + p) % p;
        int y = (id + t) % p;
        int s1 = (x & h) == 0 ? right : MPI_PROC_NULL;
        int r1 = (((x + p - 1) % p) & h) == 0 ? left : MPI_PROC_NULL;
        int s2 = (y & h) != 0 ? left : MPI_PROC_NULL;
        int r2 = (((y + 1) % p) & h) != 0 ? right : MPI_PROC_NULL;
        MPI_Request req[4];
        MPI_Irecv(u.data(), m, MPI_INT, r1, 0, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(v.data(), m, MPI_INT, r2, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Isend(cw.data(), m, MPI_INT, s1, 0, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(ccw.data(), m, MPI_INT, s2, 1, MPI_COMM_WORLD, &req[3]);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        cw.swap(u);
        ccw.swap(v);
    }
    b.swap((id & h) ? cw : ccw);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int id, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int n = argc == 1 ? p : argc - 1;
    int m = n / p;
    int ok = (p & (p - 1)) == 0 && m > 0 && n % p == 0;
    vector<int> data;
    if (id == 0 && ok) {
        data.resize(n);
        for (int i = 0; i < n; ++i) {
            if (argc == 1) {
                data[i] = n - i;
            } else {
                string_view s(argv[i + 1]);
                auto r = from_chars(s.begin(), s.end(), data[i]);
                if (r.ec != errc() || r.ptr != s.end()) ok = 0;
            }
        }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok) {
        if (id == 0) cerr << "Use a power of two ranks and an equal number of integer keys per rank.\n";
        MPI_Finalize();
        return 1;
    }

    vector<int> a(m), b(m), c(2 * static_cast<size_t>(m));
    MPI_Scatter(data.data(), m, MPI_INT, a.data(), m, MPI_INT, 0, MPI_COMM_WORLD);
    sort(a.begin(), a.end());
    for (unsigned k = 2; k <= static_cast<unsigned>(p); k *= 2) {
        for (int h = k / 2; h > 0; h /= 2) {
            exchange(a, b, id, p, h);
            merge(a.begin(), a.end(), b.begin(), b.end(), c.begin());
            bool low = ((id & k) == 0) == ((id & h) == 0);
            auto start = c.begin() + (low ? 0 : m);
            copy_n(start, m, a.begin());
        }
    }
    MPI_Gather(a.data(), m, MPI_INT, data.data(), m, MPI_INT, 0, MPI_COMM_WORLD);
    if (id == 0) {
        for (int i = 0; i < n; ++i) cout << (i ? " " : "") << data[i];
        cout << '\n';
    }
    MPI_Finalize();
}
