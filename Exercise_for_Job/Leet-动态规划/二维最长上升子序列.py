def maxEnvelopes(nums) -> int:
    if not nums:
        return 0
    sort_data = sorted(nums, key=lambda x: (x[0], -x[1]))
    data = [x[1] for x in sort_data]
    n = len(data)
    res = [1] * n
    for i in range(n):
        for j in range(i):
            if data[i] > data[j]:
                res[i] = max(res[i], res[j] + 1)
    return max(res)