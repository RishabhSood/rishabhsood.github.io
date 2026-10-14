#include <cstdio>
#include <vector>
#include <omp.h>

int main() {
    std::vector<int> values(1000);
    for (int i = 0; i < 1000; ++i) values[i] = i + 1;
    long long sum = 0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 0; i < 1000; ++i) sum += values[i];
    std::printf("threads=%d sum=%lld\n", omp_get_max_threads(), sum);
}

