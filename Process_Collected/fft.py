import numpy as np

def __bit_reverse(arr, n_bits):
    result = np.zeros(arr.shape, dtype=arr.dtype)
    for _ in range(n_bits):
        result = np.bitwise_or(np.left_shift(result, 1), np.bitwise_and(arr, 1))
        arr = np.right_shift(arr, 1)
    return result


def fft(x, pad=True, axis=0):
    if x.shape[axis] <= 1:
        return np.copy(x)

    N = x.shape[axis]
    levels = int(np.log2(N))

    if N & (N - 1) != 0:
        if pad:
            levels += 1
        N = int(2**(levels))
        if pad:
            pad = np.zeros((x.ndim, 2), dtype=np.int32)
            pad[axis, 1] = N - x.shape[axis]
            x = np.pad(x, pad)
        else:
            sl = [slice(None)] * x.ndim
            sl[axis] = slice(None, N)
            x = x[tuple(sl)]

    inds = np.arange(N)
    rev = __bit_reverse(inds, levels)
    sl = [slice(None)] * x.ndim
    sl[axis] = rev
    X = x[tuple(sl)].astype(np.complex128)

    butterflies = 2**(np.arange(levels) + 1)
    sl_half_but = [slice(None)] * X.ndim
    sl_direct = [slice(None)] * X.ndim
    for butf_s in butterflies:
        wM = np.exp(-2j * np.pi / butf_s)
        half_but = butf_s // 2
        for k in range(0, N, butf_s):
            w = 1.0
            for j in range(half_but):
                sl_half_but[axis] = k + j + half_but
                sl_direct[axis] = k + j
                temp = w * X[tuple(sl_half_but)]
                X[tuple(sl_half_but)] = X[tuple(sl_direct)] - temp
                X[tuple(sl_direct)] += temp
                w *= wM

    return X



if __name__ == "__main__":
    mine = fft(np.array([1, 2, 3, 2, 1, 2, 3, 2, 1]), axis=0) 
    mine2 = fft(np.array([[1, 2, 3, 2, 1, 2, 3, 2, 1], [1, 2, 3, 2, 1, 2, 3, 2, 1]]), axis=1) 
    nps = np.fft.fft(np.array([1, 2, 3, 2, 1, 2, 3, 2, 1]))
    npspad = np.fft.fft(np.array([1, 2, 3, 2, 1, 2, 3, 2, 1]), n=16)
    npspad = np.fft.fft(np.array([[1, 2, 3, 2, 1, 2, 3, 2, 1], [1, 2, 3, 2, 1, 2, 3, 2, 1]]), n=16, axis=1)
    print(mine.shape, mine)
    print(mine2.shape, mine2)
    print(nps.shape, nps)
    print(npspad.shape, npspad)