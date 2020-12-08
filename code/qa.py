from cw_functions import subsample, get_G


def run():
    seed = 0
    N = 16
    l = 0.3
    subsample_factor = 4

    indices = subsample(N, subsample_factor, seed)
    G = get_G(N, indices)
    print(G)


if __name__ == "__main__":
    print("4M24 - a")
    run()