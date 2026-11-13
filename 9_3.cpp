/*
Let n = 2m. Pair a[i] with a[i+m]. Put the smaller key in L[i] and the
larger in H[i]. Pick any threshold t and mark keys greater than t with 1.
On a bitonic circle the ones form one continuous arc. Let its length be k.

If k <= m, no opposite pair contains two ones. L is all zeros, and the
ones in H form one arc modulo m. If k > m, the zeros form an arc shorter
than m. H is all ones, and the zeros in L form one arc. Thus in each half
the keys above every threshold form one arc. This makes each half bitonic:
cut at a minimum and follow either side to a maximum. A drop on that path
would split the arc for a threshold between the two values.

At every threshold, L is all zeros or H is all ones. If L[i] > H[j],
choosing t = H[j] would break this rule. Hence max(L) <= min(H).

(a) A rise followed by a fall puts the marked keys in one arc at the peak.
(b) Moving the peak changes the arc length, which the proof allows.
(c) Rotating the input rotates the arc and keeps opposite positions paired.
*/

#include <algorithm>
#include <charconv>
#include <iostream>
#include <string_view>
#include <vector>

using namespace std;

bool bitonic(const vector<int>& a) {
    size_t n = a.size();
    size_t k = min_element(a.begin(), a.end()) - a.begin();
    bool down = false;
    for (size_t i = 1; i < n; ++i) {
        int x = a[(k + i - 1) % n];
        int y = a[(k + i) % n];
        if (x > y) down = true;
        if (down && x < y) return false;
    }
    return true;
}

void show(const char* name, const vector<int>& a) {
    cout << name;
    for (int x : a) cout << ' ' << x;
    cout << '\n';
}

int main(int argc, char** argv) {
    vector<int> a;
    for (int i = 1; i < argc; ++i) {
        string_view s(argv[i]);
        int x;
        auto r = from_chars(s.begin(), s.end(), x);
        if (r.ec != errc() || r.ptr != s.end()) {
            cerr << "Use integer keys.\n";
            return 1;
        }
        a.push_back(x);
    }
    if (a.empty()) a = {8, 7, 3, 0, 1, 2, 4, 9};
    size_t n = a.size();
    if (n < 2 || (n & (n - 1)) || !bitonic(a)) {
        cerr << "Use a bitonic sequence of length 2, 4, 8, ...\n";
        return 1;
    }
    size_t m = n / 2;
    vector<int> lo(m), hi(m);
    for (size_t i = 0; i < m; ++i) {
        lo[i] = min(a[i], a[i + m]);
        hi[i] = max(a[i], a[i + m]);
    }
    show("s1", lo);
    show("s2", hi);
    bool sep = *max_element(lo.begin(), lo.end()) <= *min_element(hi.begin(), hi.end());
    cout << boolalpha << "Bitonic halves: " << (bitonic(lo) && bitonic(hi)) << '\n';
    cout << "Separated: " << sep << '\n';
}
