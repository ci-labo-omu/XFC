from matplotlib import pyplot as plt

def non_dominated(F):
# X is 2d array of scores([f1, f2])
    Y = sorted(F, key=lambda x: x[0])
    # まずf1についてFをソートする．f1が小さい(優れた)ほうからf2を見て，より大きければ，その行を除く
    a = Y[0][1]
    nondom = []
    nondom.append(Y[0])
    for y in Y:
        if y[1] < a:
            print(y[1],a)
            nondom.append(y)
            a = y[1]

    return nondom

def plot_scatter(f1, f2, title=""):
    plt.title(title)
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.scatter(f1, f2)
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.show()

def plot_free(f1, f2, title=""):
    plt.title(title)
    plt.xlim(0, 5.0)
    plt.ylim(0, 5.0)
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.scatter(f1, f2)
    plt.savefig(f"{title}.png")