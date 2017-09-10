"""A test program for gradient descent
"""

train_set = [(1, 1), (2, 1), (2, 2), (3, 2)]

def partial_k(k, x, y):
    return 2 * k * x * x + 2 * b * x - 2 * x * y

def partial_b(b, x, y):
    return 2 * k * x + 2 * b - 2 * y
def loss_func(k, b, x, y):
    return (k * x + b - y) ** 2

k = 0
b = 0
lr = 0.12

tmp_loss = 0
for epoch in xrange(200):
    total_loss = 0
    cu_k = 0
    cu_b = 0
    for pairs in train_set:
        total_loss += loss_func(k, b, pairs[0], pairs[1])
        cu_k += partial_k(k, pairs[0], pairs[1])
        cu_b += partial_b(b, pairs[0], pairs[1])
    print "The updated k and b value are ", k, b, '******', epoch
    print "The average loss for this epoch is ", total_loss / len(train_set)
    if tmp_loss == 0:
        tmp_loss = total_loss
    else:
        print "The tmp loss is now ", tmp_loss
        if tmp_loss < total_loss:
            print "Try a smaller learning rate"
            break
    if total_loss / len(train_set) < 0.000001: break
    k = k - lr * cu_k / len(train_set)
    b = b - lr * cu_b / len(train_set)
