import numpy as np

class Transmitor :

    def __init__(self):
        super()


    #-------------------------------------------------------------------------------------------------#
    
    def source(self, N, p):
        """ Generates an N-bit sequence drawn from a Bernoulli stochastic process.
            @param N : Length of the bit sequence
            @param p : probability of 0
        """
        n=1
        return np.random.binomial(n, p, size=N)


    #-------------------------------------------------------------------------------------------------#

    def build_constellations(self, M):
    
        """ Builds a M-QAM constellation.
            @param M : Number of symbols.
        """
        
        # Sequential address from 0 to M-1 (1xM dimension)
        n = np.arange(0,M)
        #convert linear addresses to Gray code
        a = np.asarray([x^(x>>1) for x in n])
        #Dimension of K-Map - N x N matrix
        D = np.sqrt(M).astype(int) 
        # NxN gray coded matrix
        a = np.reshape(a,(D,D))
        # identify alternate rows
        oddRows=np.arange(start = 1, stop = D ,step=2) 
        
        # reshape to 1xM - Gray code walk on KMap
        nGray=np.reshape(a,(M)) 
        
        #element-wise quotient and remainder
        (x,y)=np.divmod(nGray,D) 
        # PAM Amplitudes 2d+1-D - real axis
        Ax=2*x+1-D 
        # PAM Amplitudes 2d+1-D - imag axis
        Ay=2*y+1-D 
        constellation = Ax+1j*Ay
        
        self.constellation = constellation

        return constellation


    #-------------------------------------------------------------------------------------------------#

    def bit_to_symb(self, b, cnt):
        """ Creates a mapping between bits sequences and symbols.
        @param b : N-bit sequence
        @param cnt : constellation
        """

        k = int(np.log2(M))

        if b.size % k != 0:
            b = np.vstack((b, np.zeros((6-b.size % k, 1), dtype=np.uint8)))

        bits = b.reshape((-1, k))

        symboles = []
        for bi in bits:
            biDec = np.packbits(np.hstack((np.zeros(8-k, dtype=np.uint8), bi)))[0]
            bin, code = self.grayCoding(biDec, M)
            symboles.append(code)

        return np.array(symboles)

    #-------------------------------------------------------------------------------------------------#

    def grayCoding(self, n, M):
        k = int(np.log2(M))

        reAxis = np.hstack((np.arange(-(np.sqrt(M)//2), 0, step=1),
                            np.arange(1, np.sqrt(M)/2+1, step=1)))
        imAxis = np.copy(reAxis)

        bin = np.unpackbits(np.array([n], dtype=np.uint8))[8-k:]

        rePart = bin[:k//2]
        imPart = bin[k//2:]

        aReBin = np.hstack((np.zeros(8-k//2, dtype=np.uint8), rePart))
        aImBin = np.hstack((np.zeros(8-k//2, dtype=np.uint8), imPart))

        aReDec = np.packbits(aReBin)
        aImDec = np.packbits(aImBin)

        reIndex = np.bitwise_xor(aReDec, aReDec//2)[0]
        imIndex = np.bitwise_xor(aImDec, aImDec//2)[0]

        return  bin, complex(reAxis[reIndex], imAxis[imIndex])
    
    #-------------------------------------------------------------------------------------------------#