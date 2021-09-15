from Utilities import size, error


def checkInterpInput(g, x):
    if size(x, 1) != g.dim:
        if size(x, 0) == g.dim:
            # Take transpose if number of input rows is same as grid dimension
            x = x.T;
        else:
            error('Input points must have the same dimension as grid!')

    return x
