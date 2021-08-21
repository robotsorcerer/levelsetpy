from utils import *
# double check this


def cellMatrixMultiply(A, B):
    if(len(A.shape)>2):
        if(len(B.shape)>2):
            # Full matrix/matrix multiplication
            sizeA = size(A)
            sizeB = size(B)

            if((sizeA != 2) or (sizeB != 2)):
                error('A and B must be cell arrays of dimension 2.')
            if(sizeA[1] != sizeB[0]):
                error('Inner dimensions of A and B must match.')

            C = np.empty((sizeA[0], sizeB[1]))
            for i in range(sizeA[0]):
                for j in range(sizeB[1]):
                    C[i,j] = A[i,0] * B[i,j]
                    for k in range(1, sizeA[1]):
                        C[i,j] += A[i,k] * B[k,j]
            return C

        elif(isnumeric(B)):
            # B will multiply every entry of A.
            scalar = B
            array = A
        else:
            error('Input B must be a numeric array or a cell matrix')

    elif(isnumeric(A)):
        if(iscell(B)):
            # A will multiply every entry of B.
            scalar = A
            array = B

        elif(isnumeric(B)):
            # Regular pointwise array multiplication (including scalar * scalar).
            C = A * B
            return C

        else:
            error('Input B must be a numeric array or a cell matrix')
    else:
        error('Input A must be a numeric array or a cell matrix')


    # If we drop through to here, one of the inputs was not a cell matrix.
    sizeArray = size(array)
    if(len(sizeArray) != 2):
        error('Cell array must be of dimension 2')

    C = cell(sizeArray)
    for i in range(sizeArray[0]):
        for j in range(sizeArray[1]):
            C[i,j] = scalar * array[i,j]

  return C
