import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import dot
from numpy.linalg import norm


from load import DataLoad
dl = DataLoad()

class Models:


    def _plot(self, scores):
        axes=[]
        fig=plt.figure(figsize=(8,8))
        for a in range(5*6):
            score = scores[a]
            axes.append(fig.add_subplot(5, 6, a+1))
            subplot_title=str(round(score[0],2)) + "/m" + str(score[2]+1)
            axes[-1].set_title(subplot_title)  
            plt.axis('off')
            plt.imshow(Image.open(score[1]))
        fig.tight_layout()
        return plt.show()


    def L2_norm(self, query, features, img_paths):
        dists = np.linalg.norm(features - query, axis=1)
        ids = np.argsort(dists)[:30]
        scores = [(dists[id], img_paths[id], id) for id in ids]

        return self._plot(scores)



    def _cos_sim(self, a, b):
        return dot(a, b) / (norm(a) * norm(b))
    

    def cos_score(self, query , features, img_paths):
        # 각 이미지의 코사인 유사도를 계산하고 정렬
        cos_scores = []
        for feature in features:
            similarity = self._cos_sim(query, feature)
            cos_scores.append(similarity)

        dists = cos_scores
        ids = np.argsort(dists)[::-1]
        scores = [(dists[id], img_paths[id], id) for id in ids]

        return self._plot(scores)

    


