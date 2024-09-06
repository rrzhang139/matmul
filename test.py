import os
import time
import numpy as np

os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"

N = 1024


def main():
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    for i in range(3):
        start = time.time()
        C = A @ B
        end = time.time()

        flops = N*N*2*N
        gflops = flops / 1e9
        print(f"{gflops:.2f} GFLOPs")
        s = end - start
        print(
            f"{(gflops / s):.2f} GFLOPs/s")


if __name__ == "__main__":
    main()
    print("=====")
