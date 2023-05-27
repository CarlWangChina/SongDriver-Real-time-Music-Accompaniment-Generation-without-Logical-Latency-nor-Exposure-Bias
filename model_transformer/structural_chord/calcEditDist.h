#pragma once
#include <tuple>
#include <vector>
namespace chcpy {

template <typename T>
inline int calcEditDist(const T& s1, const T& s2) {
    int m = s1.size();
    int n = s2.size();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));
    for (int i = 1; i <= m; i++) {  //<=说明遍历的次数是m-1+1次，即字符串的长度
        dp[i][0] = i;
    }
    for (int j = 1; j <= n; j++) {
        dp[0][j] = j;
    }
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            //if (s1[i] == s2[j])//运行结果不对，因为第i个字符的索引为i-1
            if (s1[i - 1] == s2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];  //第i行j列的步骤数等于第i-1行j-1列，因为字符相同不需什么操作，所以不用+1
            } else {
                //if (i == 1) {
                //    dp[i][j] = std::min(std::min(dp[i - 1][j] + 1, dp[i][j - 1] + 1), dp[i - 1][j - 1] + 1);  //+1表示经过了一次操作
                //} else {
                    dp[i][j] = std::min(std::min(dp[i - 1][j] + 1, dp[i][j - 1] + 1), dp[i - 1][j - 1] + 1);  //+1表示经过了一次操作
                //}
            }
        }
    }
    return dp[m][n];
}

inline float calcEditDist(const std::vector<std::tuple<int, float>>& A, const std::vector<int>& B, float act = 1.0) {
    int lenA = A.size();
    int lenB = B.size();
    int i, j;
    std::vector<std::vector<float>> dp(lenA + 1, std::vector<float>(lenB + 1));

    //初始化边界
    for (i = 1; i <= lenA; i++) {
        dp.at(i).at(0) = std::get<1>(A.at(i - 1)) + dp.at(i - 1).at(0);
    }
    for (j = 1; j <= lenB; j++) {
        dp.at(0).at(j) = j;
    }

    //dp
    for (i = 1; i <= lenA; i++) {
        for (j = 1; j <= lenB; j++) {
            //if (s1[i] == s2[j])//运行结果不对，因为第i个字符的索引为i-1
            if (std::get<0>(A.at(i - 1)) == B.at(j - 1)) {
                dp.at(i).at(j) = dp.at(i - 1).at(j - 1);  //第i行j列的步骤数等于第i-1行j-1列，因为字符相同不需什么操作，所以不用+act
            } else {
                float weight_last = std::get<1>(A.at(i - 1));
                float weight_now = i < lenA ? std::get<1>(A.at(i)) : act;
                dp.at(i).at(j) = std::min(std::min(
                                              dp.at(i - 1).at(j) + weight_last,
                                              dp.at(i).at(j - 1) + weight_now),
                                          dp.at(i - 1).at(j - 1) + weight_last) +
                                 act;  //+act表示经过了一次操作
            }
        }
    }
#ifdef MGNR_DEBUG
    printf("lenA=%d lenB=%d\n", lenA, lenB);
    for (i = 0; i <= lenA; i++) {
        for (j = 0; j <= lenB; j++) {
            printf("%f\t", dp[i][j]);
        }
        printf("\n");
    }
    printf("\n");
#endif

    return dp.at(lenA).at(lenB);
}

}  // namespace chcpy