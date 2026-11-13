/*
Section 9.1 defines A <= B by max(A) <= min(B). Take A = [1,2].
Then A <= A would require 2 <= 1. This fails, so the printed relation
is not reflexive and cannot be a partial order.

For nonempty blocks, transitivity holds: A <= B and B <= C give
max(A) <= min(B) <= max(B) <= min(C). For equal-size blocks,
A <= B and B <= A force every entry in both blocks to have the same
value. This proves antisymmetry. One fix is to allow A <= B when the
blocks are equal or max(A) <= min(B). This also gives reflexivity.
*/

#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

bool le(const vector<int>& a, const vector<int>& b) {
    return *max_element(a.begin(), a.end()) <= *min_element(b.begin(), b.end());
}

int main() {
    vector<int> a = {1, 2};
    int lo = *min_element(a.begin(), a.end());
    int hi = *max_element(a.begin(), a.end());
    cout << boolalpha;
    cout << "A = [1, 2]\n";
    cout << "A <= A requires " << hi << " <= " << lo << ".\n";
    cout << "A <= A is " << le(a, a) << ".\n";
    cout << "This counterexample breaks reflexivity.\n";
}
