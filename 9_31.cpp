/*
Let b be the key width and r the bits per pass, where 1 <= r < b.
In the printed formula T = (b/r)*2^r*(Theta(log n) + Theta(n)).
For exact full passes, minimize f(r)=2^r/r. The ratio f(r+1)/f(r)
is 2*r/(r+1). It is 1 at r=1 and greater than 1 afterwards. Hence
r=1 and r=2 tie when both are allowed and divide b. With ceil(b/r)
full-width passes, r=1 always wins or ties; r=2 ties for even b>2.

In the pseudocode the final permutation runs only once per pass. Counting
the loops gives T = ceil(b/r)*(A*2^r+B), where A is the cost of the two
collectives for one digit value and B is the cost of one permutation.
For A=log2(n), B=n, n=1024 and b=32, trying integer r gives r=5.
This second result depends on the cost model.
*/

#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>

bool num(const char* s, int& n) {
    std::istringstream in(s);
    return (in >> n) && in.eof();
}

int main(int argc, char** argv) {
    int n = 1024, b = 32;
    if ((argc != 1 && argc != 3) ||
        (argc == 3 && (!num(argv[1], n) || !num(argv[2], b))) ||
        n < 2 || b < 2 || b > 64) {
        std::cerr << "Usage: 9_31 [n bits], n >= 2 and 2 <= bits <= 64\n";
        return 1;
    }
    double lg = std::log2(n);
    std::vector<double> u(b), v(b);
    std::cout << "r  passes  printed_cost  loop_cost\n";
    for (int r = 1; r < b; ++r) {
        int k = (b + r - 1) / r;
        double d = std::ldexp(1.0, r);
        u[r] = k * d * (lg + n);
        v[r] = k * (d * lg + n);
        std::cout << r << ' ' << k << ' ' << u[r] << ' ' << v[r] << '\n';
    }
    for (int f = 0; f < 2; ++f) {
        std::cout << (f == 0 ? "Best r (printed model):" : "Best r (loop count):");
        const auto& t = f == 0 ? u : v;
        double best = *std::min_element(t.begin() + 1, t.end());
        for (int r = 1; r < b; ++r) {
            if (t[r] == best) std::cout << ' ' << r;
        }
        std::cout << '\n';
    }
}
