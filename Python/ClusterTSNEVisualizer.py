#Agent uses t-SNE to visualize generate clusters
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

class ClusterTSNEVisualizer:
    def execute(self, df):
        tsne = TSNE(n_components=3)
        tsne_result = tsne.fit_transform(df.drop(['subject_id', 'cluster'], axis=1))
        df_tsne = pd.DataFrame(tsne_result, columns=['x', 'y', 'z'])
        df_tsne['cluster'] = df['cluster']
        sns.pairplot(df_tsne, hue='cluster')
        plt.show()
