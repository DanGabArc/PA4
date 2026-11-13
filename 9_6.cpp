/*
At step j, rank r compares with r XOR 2^j. These ranks differ in exactly
one bit, so they are neighbors on the hypercube. They agree on the sort
direction and have opposite bit j. One keeps the smaller value and the
other keeps the larger. In an ascending group the lower rank keeps the
smaller value. In a descending group it keeps the larger one.

A single key is sorted. Suppose adjacent groups of length 2^i are sorted
in opposite directions. Joining them gives a bitonic sequence. The first
split separates its low and high halves. Each half is still bitonic, so
the next splits sort both halves in the required direction. This proves
the next stage by induction. The final group is ascending, so rank r
holds the key at sorted position r, starting from zero.
*/

#include <algorithm>
#include <charconv>
#include <iostream>
#include <mpi.h>
#include <string_view>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int id, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    vector<int> a;
    int ok = (p & (p - 1)) == 0 && (argc == 1 || argc - 1 == p);
    if (id == 0 && ok) {
        a.resize(p);
        for (int i = 0; i < p; ++i) {
            if (argc == 1) {
                a[i] = p - i;
            } else {
                string_view s(argv[i + 1]);
                auto r = from_chars(s.begin(), s.end(), a[i]);
                if (r.ec != errc() || r.ptr != s.end()) ok = 0;
            }
        }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok) {
        if (id == 0) cerr << "Use 1, 2, 4, 8, ... ranks and one integer per rank, or no arguments.\n";
        MPI_Finalize();
        return 1;
    }

    int x, y;
    MPI_Scatter(a.data(), 1, MPI_INT, &x, 1, MPI_INT, 0, MPI_COMM_WORLD);
    for (unsigned k = 2; k <= static_cast<unsigned>(p); k *= 2) {
        for (int h = k / 2; h > 0; h /= 2) {
            int dst = id ^ h;
            MPI_Sendrecv(&x, 1, MPI_INT, dst, 0, &y, 1, MPI_INT, dst, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            bool low = ((id & k) == 0) == ((id & h) == 0);
            x = low ? min(x, y) : max(x, y);
        }
    }
    MPI_Gather(&x, 1, MPI_INT, a.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (id == 0) {
        for (int i = 0; i < p; ++i) cout << (i ? " " : "") << a[i];
        cout << '\n';
    }
    MPI_Finalize();
}
