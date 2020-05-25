import numpy as np




n = 32
k = 16
nlayer = 5
from Channel import get_dimen,get_I,convsnrstd,DataGeneratorRM_AWGN,get_frozen_index
dataset = DataGeneratorRM_AWGN(n,k)
snr = np.array([6])
gen = dataset.generator_train_data(convsnrstd(snr), batch_size= 1000, input_type='llr')


frozen_bits = np.array(get_frozen_index())
np.set_printoptions(linewidth=np.nan)

def decode ( input ):

    llr = np.zeros((nlayer+ 1 ,n))
    beta = np.zeros ((nlayer + 1 ,n))
    llr[nlayer] = input
    for i in range(n):

        str= np.binary_repr(i,nlayer)
        str = str[::-1]
	for j in range(nlayer):
            nj = nlayer - j - 1
            mold1 = np.ones (2**(nj))
            mold2 = np.zeros(2**(nj))
            if str[nj] == '0':

                mold = np.concatenate((mold1,mold2))
                mold = np.tile(mold,2**(nlayer - nj - 1 ))
                vec =np.copy( llr [nj + 1  ])
                vec = np.clip(vec, a_min = -10, a_max = 10)
                vec = np.tanh(vec/2)
                vec2 = np.roll(vec,-2**(nj))
                vec = vec * vec2

                vec = 2 *  np.arctanh(vec)
                vec *= mold
                llr[nj] = llr[nj] * ( 1 - mold ) + vec

            else :

                mold = np.concatenate((mold2,mold1))
                mold = np.tile(mold,2**(nlayer-nj-1 ))
                vec =np.copy( llr [nj + 1  ])
                s = beta[nj]
                s = 1 - 2 *s
                vec2 = vec * s
                vec2 = np.roll(vec2, 2**(nj ))
                vec = vec + vec2
                vec = vec * mold
                llr[nj] = llr[nj] * ( 1 - mold ) + vec

        if llr[0][i] >= 0  :
            beta[0][i] = 0
        else:
            beta[0][i] = 1
        b = np.copy(beta[0])
        b = b * frozen_bits
        beta [0] = b
        for j in range(nlayer):
            mold1 = np.ones (2**(j))
            mold2 = np.zeros(2**(j))
            mold = np.concatenate((mold1,mold2))
            mold = np.tile(mold,2**(nlayer - j - 1 ))
            vec = np.copy(beta[j])
            vec2 = np.roll(vec,-2**j)
            vec = np.logical_xor(vec,vec2) * 1.0
            vec *= mold
            beta[j + 1 ] = (1-mold)* beta[j] + vec

    return beta[0]


def batch_trial( inp , targ) :
    BER = 0
    for i in range(inp.shape[0]):
        pred = decode(inp[i])
        trg = targ[i]
        BER_ =np.sum(np.not_equal(pred, trg))/(pred.shape[0])

        BER += BER_
    BER/= inp.shape[0]
    return BER
