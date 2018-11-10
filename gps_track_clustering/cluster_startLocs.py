from gps_track_clustering.cluster import Cluster
from sklearn.cluster import DBSCAN


class ClusterStartLocs(Cluster):

    _MIN_SAMPLES = 7        # Minimum size of cluster
    _EPS         = 0.002    # Distance between points beeing clustered / tolerance. 0.0005deg ~ 55m

    def __init__(self, summary):
        Cluster.__init__(self, summary)

        # CONFIG
        self.desc = 'ClusterStartLocs'
        self.set_config()

    def set_config(self, min_samples=_MIN_SAMPLES, eps=_EPS):
        self.min_samples = min_samples
        self.eps         = eps

    def _apply_cluster_algo(self):
        labels, n_clusters, cluster_ids = self._train_predict_DBSCAN()
        return labels, n_clusters, cluster_ids

    def _train_predict_DBSCAN(self):
        # Select features
        features = self.summary[['startpos_lat', 'startpos_lon']]

        # Apply DBSCAN algorithm
        print('Config: eps={}  /  min_samples={}'.format(self.eps, self.min_samples))
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        db.fit(features)

        # Evaluate results
        # core_samples    = db.core_sample_indices_
        labels          = db.labels_
        n_clusters      = len(set(labels)) - (1 if -1 in labels else 0)
        cluster_ids     = set(labels)

        return labels, n_clusters, cluster_ids
