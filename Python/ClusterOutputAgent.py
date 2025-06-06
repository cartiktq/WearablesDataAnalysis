#Agent do describe the output of the clustering agent
class ClusterOutputAgent:
    def execute(self, clustered_df):
        return clustered_df.groupby('cluster').describe(include='all')
