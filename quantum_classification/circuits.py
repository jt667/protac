import pennylane as qml
from numpy.polynomial import polynomial


def parameter_count(circuit_num, num_wires):
    # Read polynomial coefficient data
    with open("parameter_polynomials.txt") as f:
        content = f.readlines()
    # Convert strings to lists of coefficients [a0, a1, a2, ...]
    param_polynomials = []
    for line in content:
        coef_str = line.rstrip().split(",")
        coef_flt = [float(x) for x in coef_str]
        param_polynomials.append(coef_flt)

    # Return the evaluated polynomial p(numWires) as no. params needed
    return int(polynomial.polyval(num_wires, param_polynomials[circuit_num - 1]))


def circuit1(params, x):
    n = x.size
    # Requires 2n parameters
    qml.templates.AngleEmbedding(x, wires=[i for i in range(n)])
    # RX, RZ block
    for i in range(n):
        qml.RX(params[2 * i], wires=i)
        qml.RZ(params[2 * i + 1], wires=i)
    measurements = [qml.expval(qml.PauliZ(i)) for i in range(n)]
    return measurements


def circuit2(params, x):
    n = x.size
    # Requires 2n parameters
    qml.templates.AngleEmbedding(x, wires=[i for i in range(n)])
    # RX, RZ block
    for i in range(n):
        qml.RX(params[2 * i], wires=i)
        qml.RZ(params[2 * i + 1], wires=i)
    # CNOT chain
    qml.broadcast(qml.CNOT, [i for i in range(n - 1, -1, -1)], "chain")
    measurements = [qml.expval(qml.PauliZ(i)) for i in range(n)]
    return measurements


def circuit3(params, x):
    n = x.size
    # Requires 3n-1 parameters
    qml.templates.AngleEmbedding(x, wires=[i for i in range(n)])
    # RX, RZ block
    for i in range(n):
        qml.RX(params[2 * i], wires=i)
        qml.RZ(params[2 * i + 1], wires=i)
    # CRZ Chain
    for i in range(n - 1):
        qml.CRZ(params[2 * n + i], wires=[n - 1 - i, n - 2 - i])
    measurements = [qml.expval(qml.PauliZ(i)) for i in range(n)]
    return measurements


def circuit4(params, x):
    n = x.size
    # Requires 3n-1 parameters
    qml.templates.AngleEmbedding(x, wires=[i for i in range(n)])
    # RX, RZ block
    for i in range(n):
        qml.RX(params[2 * i], wires=i)
        qml.RZ(params[2 * i + 1], wires=i)
    # CRX Chain
    for i in range(n - 1):
        qml.CRX(params[2 * n + i], wires=[n - 1 - i, n - 2 - i])
    measurements = [qml.expval(qml.PauliZ(i)) for i in range(n)]
    return measurements


def circuit5(params, x):
    n = x.size
    # Requires n**2 + 3n parameters
    qml.templates.AngleEmbedding(x, wires=[i for i in range(n)])
    # RX, RZ block
    for i in range(n):
        qml.RX(params[2 * i], wires=i)
        qml.RZ(params[2 * i + 1], wires=i)
    # CRZ connect to all
    index = 2 * n
    for i in range(n):
        for j in range(n):
            if not i == j:
                qml.CRZ(params[index], wires=[n - 1 - i, n - 1 - j])
                index += 1
    for i in range(n):
        qml.RX(params[2 * i + index], wires=i)
        qml.RZ(params[2 * i + 1 + index], wires=i)
    measurements = [qml.expval(qml.PauliZ(i)) for i in range(n)]
    return measurements


def circuit6(params, x):
    n = x.size
    # Requires n**2 + 3n parameters
    qml.templates.AngleEmbedding(x, wires=[i for i in range(n)])
    # RX, RZ block
    for i in range(n):
        qml.RX(params[2 * i], wires=i)
        qml.RZ(params[2 * i + 1], wires=i)
    # CRX connect to all
    index = 2 * n
    for i in range(n):
        for j in range(n):
            if not i == j:
                qml.CRX(params[index], wires=[n - 1 - i, n - 1 - j])
                index += 1
    for i in range(n):
        qml.RX(params[2 * i + index], wires=i)
        qml.RZ(params[2 * i + 1 + index], wires=i)
    measurements = [qml.expval(qml.PauliZ(i)) for i in range(n)]
    return measurements


def circuit7(params, x):
    n = x.size
    # Requires 9/8 n**2 + n/4 parameters, n even
    qml.templates.AngleEmbedding(x, wires=[i for i in range(n)])
    index = 0
    for i in range(int(n / 2)):
        # RX, RZ blocks
        for j in range(n):
            qml.RX(params[index], wires=j)
            index += 1
            qml.RZ(params[index], wires=j)
            index += 1
        # RZ pyramid
        for j in range(int(n / 2) - i):
            qml.CRZ(params[index], wires=[i + 2 * j + 1, i + 2 * j])
            index += 1

    measurements = [qml.expval(qml.PauliZ(i)) for i in range(n)]
    return measurements


def circuit8(params, x):
    n = x.size
    # Requires 9/8 n**2 + n/4 parameters, n even
    qml.templates.AngleEmbedding(x, wires=[i for i in range(n)])
    index = 0
    for i in range(int(n / 2)):
        # RX, RZ blocks
        for j in range(n):
            qml.RX(params[index], wires=j)
            index += 1
            qml.RZ(params[index], wires=j)
            index += 1
        # RX pyramid
        for j in range(int(n / 2) - i):
            qml.CRX(params[index], wires=[i + 2 * j + 1, i + 2 * j])
            index += 1

    measurements = [qml.expval(qml.PauliZ(i)) for i in range(n)]
    return measurements


def circuit9(params, x):
    n = x.size
    # Requires n parameters
    qml.templates.AngleEmbedding(x, wires=[i for i in range(n)])
    # H block
    qml.broadcast(qml.Hadamard, [i for i in range(n)], "single")
    # CZ chain
    for i in range(n - 1):
        qml.CZ(wires=[n - 1 - i, n - 2 - i])
    # RX block
    for i in range(n):
        qml.RX(params[i], wires=[i])

    measurements = [qml.expval(qml.PauliZ(i)) for i in range(n)]
    return measurements


def circuit10(params, x):
    n = x.size
    # Requires 2n parameters
    qml.templates.AngleEmbedding(x, wires=[i for i in range(n)])
    # RY block
    for i in range(n):
        qml.RY(params[i], wires=i)
    # CZ ring
    for i in range(n - 1):
        qml.CZ(wires=[n - 1 - i, n - 2 - i])
    qml.CZ(wires=[0, n - 1])
    # RY block
    for i in range(n):
        qml.RY(params[n + i], wires=i)

    measurements = [qml.expval(qml.PauliZ(i)) for i in range(n)]
    return measurements


def circuit11(params, x):
    n = x.size
    # Requires 1/2 n**2 + n, n even
    qml.templates.AngleEmbedding(x, wires=[i for i in range(n)])
    index = 0
    for i in range(int(n / 2)):
        for j in range(n - 2 * i):
            # RY, RZ pyramid block
            qml.RY(params[index], wires=i + j)
            index += 1
            qml.RZ(params[index], wires=i + j)
            index += 1
        # CNOT pyramid block
        for j in range(int(n / 2) - i):
            qml.CNOT(wires=[i + 2 * j + 1, i + 2 * j])

    measurements = [qml.expval(qml.PauliZ(i)) for i in range(n)]
    return measurements


def circuit12(params, x):
    n = x.size
    # Requires 1/2 n**2 + n, n even
    qml.templates.AngleEmbedding(x, wires=[i for i in range(n)])
    index = 0
    for i in range(int(n / 2)):
        for j in range(n - 2 * i):
            # RY, RZ pyramid block
            qml.RY(params[index], wires=i + j)
            index += 1
            qml.RZ(params[index], wires=i + j)
            index += 1
        # CNOT pyramid block
        for j in range(int(n / 2) - i):
            qml.CZ(wires=[i + 2 * j + 1, i + 2 * j])

    measurements = [qml.expval(qml.PauliZ(i)) for i in range(n)]
    return measurements


def circuit13(params, x):
    n = x.size
    # Requires 4n parameters
    qml.templates.AngleEmbedding(x, wires=[i for i in range(n)])
    index = 0
    for i in range(n):
        # RY block
        qml.RY(params[index], wires=i)
        index += 1

    # CRZ blocks
    qml.CRZ(params[index], wires=[n - 1, 0])
    index += 1

    for i in range(n - 1):
        qml.CRZ(params[index], wires=[n - 2 - i, n - 1 - i])
        index += 1

    for i in range(n):
        # RY block
        qml.RY(params[index], wires=i)
        index += 1

    # CRZ blocks
    qml.CRZ(params[index], wires=[n - 1, n - 2])
    index += 1
    qml.CRZ(params[index], wires=[0, n - 1])
    index += 1

    for i in range(n - 2):
        qml.CRZ(params[index], wires=[i + 1, i])
        index += 1

    measurements = [qml.expval(qml.PauliZ(i)) for i in range(n)]
    return measurements


def circuit14(params, x):
    n = x.size
    # Requires 4n parameters
    qml.templates.AngleEmbedding(x, wires=[i for i in range(n)])
    index = 0
    for i in range(n):
        # RY block
        qml.RY(params[index], wires=i)
        index += 1

    # CRX ring
    qml.CRX(params[index], wires=[n - 1, 0])
    index += 1

    for i in range(n - 1):
        qml.CRX(params[index], wires=[n - 2 - i, n - 1 - i])
        index += 1

    for i in range(n):
        # RY block
        qml.RY(params[index], wires=i)
        index += 1

    # CRX ring
    qml.CRX(params[index], wires=[n - 1, n - 2])
    index += 1
    qml.CRX(params[index], wires=[0, n - 1])
    index += 1

    for i in range(n - 2):
        qml.CRX(params[index], wires=[i + 1, i])
        index += 1

    measurements = [qml.expval(qml.PauliZ(i)) for i in range(n)]
    return measurements


def circuit15(params, x):
    n = x.size
    # Requires 2n parameters
    qml.templates.AngleEmbedding(x, wires=[i for i in range(n)])
    index = 0
    for i in range(n):
        # RY block
        qml.RY(params[index], wires=i)
        index += 1

    # CNOT ring
    qml.CNOT(wires=[n - 1, 0])

    for i in range(n - 1):
        qml.CNOT(wires=[n - 2 - i, n - 1 - i])

    for i in range(n):
        # RY block
        qml.RY(params[index], wires=i)
        index += 1

    # CNOT ring
    qml.CNOT(wires=[n - 1, n - 2])
    qml.CNOT(wires=[0, n - 1])
    
    for i in range(n - 2):
        qml.CNOT(wires=[i + 1, i])

    measurements = [qml.expval(qml.PauliZ(i)) for i in range(n)]
    return measurements


def circuit16(params, x):
    n = x.size
    # Requires 1/8 n**2 + 9n/4 parameters, n even
    qml.templates.AngleEmbedding(x, wires=[i for i in range(n)])
    index = 0
    # RX, RZ block
    for j in range(n):
        qml.RX(params[index], wires=j)
        index += 1
        qml.RZ(params[index], wires=j)
        index += 1
    for i in range(int(n / 2)):

        # RZ pyramid
        for j in range(int(n / 2) - i):
            qml.CRZ(params[index], wires=[i + 2 * j + 1, i + 2 * j])
            index += 1

    measurements = [qml.expval(qml.PauliZ(i)) for i in range(n)]
    return measurements


def circuit17(params, x):
    n = x.size
    # Requires 1/8 n**2 + 9n/4 parameters, n even
    qml.templates.AngleEmbedding(x, wires=[i for i in range(n)])
    index = 0
    # RX, RZ block
    for j in range(n):
        qml.RX(params[index], wires=j)
        index += 1
        qml.RZ(params[index], wires=j)
        index += 1
    for i in range(int(n / 2)):

        # RX pyramid
        for j in range(int(n / 2) - i):
            qml.CRX(params[index], wires=[i + 2 * j + 1, i + 2 * j])
            index += 1

    measurements = [qml.expval(qml.PauliZ(i)) for i in range(n)]
    return measurements


def circuit18(params, x):
    n = x.size
    # Requires 3n parameters
    qml.templates.AngleEmbedding(x, wires=[i for i in range(n)])
    index = 0
    # RX, RZ block
    for i in range(n):
        qml.RX(params[index], wires=i)
        index += 1
        qml.RZ(params[index], wires=i)
        index += 1

    # CRZ ring
    qml.CRZ(params[index], wires=[n - 1, 0])
    index += 1

    for i in range(n - 1):
        qml.CRZ(params[index], wires=[n - 2 - i, n - 1 - i])
        index += 1

    measurements = [qml.expval(qml.PauliZ(i)) for i in range(n)]
    return measurements


def circuit19(params, x):
    n = x.size
    # Requires 3n parameters
    qml.templates.AngleEmbedding(x, wires=[i for i in range(n)])
    index = 0

    # RX, RZ block
    for i in range(n):
        qml.RX(params[index], wires=i)
        index += 1
        qml.RZ(params[index], wires=i)
        index += 1

    # CRX ring
    qml.CRX(params[index], wires=[n - 1, 0])
    index += 1

    for i in range(n - 1):
        qml.CRX(params[index], wires=[n - 2 - i, n - 1 - i])
        index += 1

    return [qml.expval(qml.PauliZ(i)) for i in range(n)]
