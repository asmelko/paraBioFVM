

# Thomas algorithm
# math ops: n * (4 FMA + 2 DIV)
# memory ops: forward: n * (2 WRITE + 4 READ), backward: n * (1 WRITE + 2 READ)
# arithmetic intensity: 14ops / 9mem
def thomas(a, b, c, d):
    n = len(d)

    c_ = [c[0] / b[0]]
    d_ = [d[0] / b[0]]
    for i in range(1, n):
        c_.append(c[i] / (b[i] - a[i] * c_[i - 1]))
        d_.append((d[i] - a[i] * d_[i - 1]) / (b[i] - a[i] * c_[i - 1]))

    x = [0] * n
    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]

    return x

# in forward substitution, we can reuse all the memory?
def cyclic_reduction(a, b, c, d):
    n = len(d)

    s = 1
    p = 2
    while p <= n:
        for i in range(p - 1, n, p):
            w1 = -a[i] / b[i - s]
            a[i] = a[i - s] * w1
            b[i] += w1 * c[i - s]
            d[i] += w1 * d[i - s] 
            if i + s < n:
                w2 = -c[i] / b[i + s]
                c[i] = c[i + s] * w2
                b[i] += w2 * a[i + s]
                d[i] += w2 * d[i + s]
        s *= 2
        p *= 2
    
    while p > 1:
        for i in range(s - 1, n, p):
            w1 = 0
            w2 = 0
            if i - s >= 0:
                w1 = a[i] * d[i - s]
            if i + s < n:
                w2 = c[i] * d[i + s]
            d[i] = (d[i] - w1 - w2) / b[i]
        s //= 2
        p //= 2

    return d

def main():
    n = 16

    a = [1] * n # starts from a[1]
    a[0] = 0

    b = [6] * n

    c = [1] * n
    c[-1] = 0

    d = [1] * n

    x = thomas(a, b, c, d)
    print(x)

    x = cyclic_reduction(a, b, c, d)
    print(x)

if __name__ == '__main__':
    main()