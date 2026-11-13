/*
Let q = sqrt(p), b = n/q and w = log2(7). A process needs q pairs of
b by b blocks. It multiplies each pair using Strassen and adds the results.
This takes Theta(q*b^w) local work. Gathering A across a row and B down
a column costs 2*ts*log2(q) + 2*tw*(q-1)*b^2 for power-of-two q.

T = Theta(q*b^w) + 2*ts*log2(q) + 2*tw*(q-1)*b^2.
Sequential Strassen takes Theta(n^w). Even with free communication,
p*T / n^w grows as p^((3-w)/2). Since w < 3, the algorithm is not
cost-optimal as the number of processes grows.
*/

#include <mpi.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using Mat = std::vector<double>;

Mat tile(const Mat &a, int n, int i, int j, int h) {
    Mat t(h * h, 0);
    for (int u = 0; u < h && i + u < n; ++u)
        for (int v = 0; v < h && j + v < n; ++v)
            t[u * h + v] = a[(i + u) * n + j + v];
    return t;
}

Mat add(const Mat &a, const Mat &b, int s = 1) {
    Mat t(a.size());
    for (size_t i = 0; i < t.size(); ++i) t[i] = a[i] + s * b[i];
    return t;
}

Mat mul(const Mat &a, const Mat &b, int n) {
    if (n == 1) return {a[0] * b[0]};
    int h = n / 2;
    Mat a0 = tile(a, n, 0, 0, h), a1 = tile(a, n, 0, h, h);
    Mat a2 = tile(a, n, h, 0, h), a3 = tile(a, n, h, h, h);
    Mat b0 = tile(b, n, 0, 0, h), b1 = tile(b, n, 0, h, h);
    Mat b2 = tile(b, n, h, 0, h), b3 = tile(b, n, h, h, h);
    Mat m1 = mul(add(a0, a3), add(b0, b3), h);
    Mat m2 = mul(add(a2, a3), b0, h);
    Mat m3 = mul(a0, add(b1, b3, -1), h);
    Mat m4 = mul(a3, add(b2, b0, -1), h);
    Mat m5 = mul(add(a0, a1), b3, h);
    Mat m6 = mul(add(a2, a0, -1), add(b0, b1), h);
    Mat m7 = mul(add(a1, a3, -1), add(b2, b3), h);
    Mat c(n * n);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < h; ++j) {
            int k = i * h + j;
            c[i * n + j] = m1[k] + m4[k] - m5[k] + m7[k];
            c[i * n + h + j] = m3[k] + m5[k];
            c[(h + i) * n + j] = m2[k] + m4[k];
            c[(h + i) * n + h + j] = m1[k] - m2[k] + m3[k] + m6[k];
        }
    }
    return c;
}

Mat pack(const Mat &a, int n, int q, int h) {
    Mat t;
    t.reserve(size_t(q) * q * h * h);
    for (int i = 0; i < q; ++i) {
        for (int j = 0; j < q; ++j) {
            Mat b = tile(a, n, i * h, j * h, h);
            t.insert(t.end(), b.begin(), b.end());
        }
    }
    return t;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int r, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int q = static_cast<int>(std::sqrt(p)), n = 4, ok = q * q == p;
    Mat a, b;
    if (r == 0) {
        if (!ok) {
            std::cerr << "Use a square number of processes.\n";
        } else if (argc == 1) {
            a = {1,2,0,3, -1,1,4,0, 2,0,1,-1, 0,3,2,2};
            b = {2,0,1,-1, 0,1,-2,3, 3,2,0,1, 1,0,2,1};
        } else if (argc == 2 && std::string(argv[1]) == "--stdin"
                   && std::cin >> n && n > 0 && n <= 16384) {
            a.resize(n * n);
            b.resize(n * n);
            for (double &v : a) ok &= bool(std::cin >> v) && std::isfinite(v);
            for (double &v : b) ok &= bool(std::cin >> v) && std::isfinite(v);
        } else {
            ok = 0;
        }
        if (q * q == p && !ok) std::cerr << "Use --stdin with n, then two n by n matrices.\n";
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok) {
        MPI_Finalize();
        return 1;
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int h = 1;
    while (h < (n + q - 1) / q) h *= 2;
    int m = h * h;
    MPI_Comm row, col;
    MPI_Comm_split(MPI_COMM_WORLD, r / q, r % q, &row);
    MPI_Comm_split(MPI_COMM_WORLD, r % q, r / q, &col);
    Mat sa, sb;
    if (r == 0) {
        sa = pack(a, n, q, h);
        sb = pack(b, n, q, h);
    }
    Mat x(m), y(m), u(size_t(q) * m), v(size_t(q) * m), c(m, 0);
    MPI_Scatter(sa.data(), m, MPI_DOUBLE, x.data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(sb.data(), m, MPI_DOUBLE, y.data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Allgather(x.data(), m, MPI_DOUBLE, u.data(), m, MPI_DOUBLE, row);
    MPI_Allgather(y.data(), m, MPI_DOUBLE, v.data(), m, MPI_DOUBLE, col);
    for (int k = 0; k < q; ++k) {
        auto ai = u.begin() + size_t(k) * m;
        auto bi = v.begin() + size_t(k) * m;
        Mat t = mul(Mat(ai, ai + m), Mat(bi, bi + m), h);
        for (int i = 0; i < m; ++i) c[i] += t[i];
    }
    Mat out(r == 0 ? size_t(p) * m : 0);
    MPI_Gather(c.data(), m, MPI_DOUBLE, out.data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (r == 0) {
        std::cout.precision(std::numeric_limits<double>::max_digits10);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                size_t k = size_t((i / h) * q + j / h) * m + (i % h) * h + j % h;
                std::cout << (j ? " " : "") << out[k];
            }
            std::cout << '\n';
        }
    }
    MPI_Comm_free(&row);
    MPI_Comm_free(&col);
    MPI_Finalize();
}
