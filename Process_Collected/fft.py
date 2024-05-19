import numpy as np

def __bit_reverse(arr, levels):
    result = np.zeros(arr.shape, dtype=arr.dtype)
    for _ in range(levels):
        result = np.bitwise_or(np.left_shift(result, 1), np.bitwise_and(arr, 1))
        arr = np.right_shift(arr, 1)
    return result


def fft(x):
    N = len(x)
    if N & (N - 1) != 0:
        raise Exception("Length of the signal must be a power of two.")

    if len(x) <= 1:
        return np.copy(x)

    levels = int(np.log2(N))

    inds = np.arange(N)
    rev = __bit_reverse(inds, levels)
    X = x[rev].astype(np.complex128)

    butterflies = 2**(np.arange(levels) + 1)
    for butf_s in butterflies:
        wM = np.exp(-2j * np.pi / butf_s)
        half_but = butf_s // 2
        for k in range(0, N, butf_s):
            w = 1.0
            for j in range(half_but):
                temp = w * X[k + j + half_but] 
                X[k + j + half_but] = X[k + j] - temp
                X[k + j] += temp
                w *= wM

    return X
        


if __name__ == "__main__":
    mine = fft(np.array([1, 2, 3, 2, 1, 2, 3, 2])) 
    nps = np.fft.fft((np.array([1, 2, 3, 2, 1, 2, 3, 2])))
    print(mine.shape, mine)
    print(nps.shape, nps)
