# %%
import numpy as np

"""隠れマルコフモデル"""
class HMM():
    """コンストラクタ"""
    def __init__(self, A, B, row, data, category):
        # パラメータ
        self._A = A
        self._B = B
        self._row = row
        # データ
        self._data = data
        # ステップ数
        self._step = len(self._data)
        # サイコロの種類数
        self._cat = category

        # 前向き確率α(時刻t,カテゴリc)
        self._alpha = None
        # 後ろ向き確率β(時刻t, カテゴリc)
        self._beta = None
        # 時刻tにおける状態st
        self._psi = None
        self._psi_vec = None
        # 確率P(X)
        self._prob_x = None
        # 確率P(xt,st)
        self._prob_xs = None
    
    """ビタービアルゴリズム"""
    def viterbi_algorithm(self):
        # 初期化
        x1 = int(self._data[0])
        self._psi = np.zeros((self._step, self._cat))
        self._psi[0,:] = self._row * self._B[:,x1]
        self._psi_vec = np.zeros((self._step, self._cat))
        self._psi_vec[0,:] += 1

        # 再帰的計算
        for t in range(1, self._step):
            xt = int(self._data[t])
            for j in range(self._cat):
                self._psi[t,j] = max(self._psi[t-1,:] * self._A[:,j]) * self._B[j,xt]
                self._psi_vec[t,j] = np.argmax(self._psi[t-1,:] * self._A[:,j]) + 1
        
        print(f"Ψt(j):\n{self._psi}\n")

        print(f"Ψt(j) Vector:\n{self._psi_vec}\n")

        # 確率P(x,s*)
        prob_xs = max(self._psi[self._step-1, :])

        print(f"P(x,s*):\n{prob_xs}\n")

        # 終了
        state = np.zeros(self._step)
        n_index = np.argmax(self._psi[self._step-1, :])
        state[self._step-1] = n_index

        # 系列の復元
        for t in range(self._step-2, -1, -1):
            n_index = self._psi_vec[t, int(state[t+1])]
            state[t] = n_index
        
        print(f"状態系列s:{state}")

if __name__ == '__main__':
    A:list=np.loadtxt('../data/A.txt')
    B:list=np.loadtxt('../data/B.txt')
    row:list=np.loadtxt('../data/row.txt')
    data:list=np.loadtxt('../data/data.txt')
    category:int=3

    hmm = HMM(A, B, row, data, category)
    hmm.viterbi_algorithm()
# %%
