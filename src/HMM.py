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
        # 確率P(X)
        self._prob_x = None
        # 確率P(xt,st)
        self._prob_xs = None

    """フォワードアルゴリズム"""
    def forward_algorithm(self):
        # 初期化
        self._alpha = np.zeros(( self._step, self._cat ))
        x1 = int(self._data[0])
        self._alpha[0, :] = self._row * self._B[:,x1].T

        # 時刻t=0 -> nまで計算
        for t in range(1, self._step):
            # 時刻tにおける観測結果xを取得
            xt= int(self._data[t])
            for j in range(self._cat):
                for i in range(self._cat):
                    self._alpha[t,j] += self._alpha[t-1,i] * self._A[i,j]
                self._alpha[t,j] *= self._B[j,xt]
        
        print(f"前向き確率α:\n{self._alpha}")
        
        # t=self._stepにおけるサイコロの出目の確率P(X)
        self._prob_x = np.sum(self._alpha, axis=1)[self._step-1]
        print(f"\n時刻t={self._step}におけるサイコロの出目の確率P(X):{self._prob_x}")
    
    """バックワードアルゴリズム"""
    def backward_algorithm(self):
        # ステップ数t,サイコロの種類c
        self._beta = np.zeros((self._step, self._cat))
        self._beta[self._step-1,:]+=1
        for t in range(self._step-2, -1, -1):
            xt1 = int(data[t])
            for j in range(self._cat):
                for i in range(self._cat):
                    self._beta[t,j] += self._A[i,j] * self._B[j, xt1] * self._beta[t+1,j]
        print(f"\n後ろ向き確率β:\n{self._beta}")
    
    """時刻tにおけるサイコロの目がxtのとき、サイコロの種類がstである確率P(xt,st)を計算"""
    def calc_prob(self):
        # 時刻tにおけるサイコロの目がxtのとき、サイコロの種類がstである確率P(xt,st)を計算
        self._prob_xs = self._alpha * self._beta
        print(f"\n時刻tにおけるサイコロの目がxtのとき、サイコロの種類がstである確率P(xt,st):\n{self._prob_xs}")

        """以下は、状態遷移の候補で各時刻tで一番高い確率だった状態を選んでいる"""
        state = []
        for s in range(self._step):
            state.append(np.argmax(self._prob_xs[s]))
        
        print(f"\n各時刻tにおける状態st:{state}")

if __name__ == '__main__':
    A:list=np.loadtxt('./data/A.txt')
    B:list=np.loadtxt('./data/B.txt')
    row:list=np.loadtxt('./data/row.txt')
    data:list=np.loadtxt('./data/data.txt')
    category:int=3

    hmm = HMM(A, B, row, data, category)
    hmm.forward_algorithm()
    hmm.backward_algorithm()
    hmm.calc_prob()