from gps_track_clustering.cluster import Cluster
from sklearn.cluster import DBSCAN
from geojson import Feature, Point, FeatureCollection
import gpxpy.gpx
import gps_track_clustering.geo
import pickle
import pandas as pd
import itertools
import math


class ClusterTrackSimilarity(Cluster):
    """ This subclass extends the abstract class Cluster()
        Clustering Method:
            The DBscan algorithm is applied with a custom distance metric. Instead of
            utilizing the euclidean norm, an adapted version of the Needleman–Wunsch
            algorithm is used as a custom GPS track similarity metric in the distance matrix.
            The idea and implementation of the adapted Needleman–Wunsch algorithm came
            from https://github.com/jonblack/cmpgpx.
            This similarity metric uses several parameters. Before 2 tracks are compared, both
            are interpolated to have a similar step size (_INTP_DIST). Afterwards both tracks are
            cut and shifted in a way to optimally reduce the _GAP_PANALTY. To reduce the computationl
            effort, only those tracks are compared, which have close start locations (_IGNORE_DIST).
            When the distance matrix is precomputed, the DBScan algorithm determines the cluster.
            Here, _EPS indicated how close the locations need to be to each other to form one cluster.
            _MIN_SAMPLES defines the smallest cluster.


    Args:
        tracks_gpx (list):       Dict containing lists of gpxpy.gpx objects, each representing one track
        summary (pd.DataFrame):  Each row in the DataFrame represents one activity.
                                 Mandatory columns are: ['id', 'startpos_lat', 'startpos_lon',
                                     'distance'  'duration'  'elevation_gain']
        dir_cache (str):         path to caching dir. None mean no caching.
    """

    # Cluster DBSCAN
    _MIN_SAMPLES = 5         # Minimum size of cluster
    _EPS         = 0.05      # Max distance between two points to be one cluster

    # Cluster Similarity
    _INTP_DIST   = 30        # [meter], Distance between 2 track points, will be interpolated to this distance
    _GAP_PANALTY = -30       # [meter], Panalty distance between two track points
    _IGNORE_DIST = 500       # [meter], distance between tracks for been not compared

    _IGNORE_SIM  = 0.70      # [percent], Similarity between two tracks below this value will be set to zero

    def __init__(self, tracks_gpx, summary, dir_cache=None):
        Cluster.__init__(self, summary)

        self.tracks_gpx = tracks_gpx
        self.dir_cache = dir_cache

        # CONFIG
        self.desc = 'ClusterTrackSimilarity'
        self.set_config()

        self.df_simMatrix = None

    def set_config(self, min_samples=_MIN_SAMPLES, eps=_EPS, intp_dist=_INTP_DIST,
                   gap_penalty=_GAP_PANALTY, ignore_dist=_IGNORE_DIST, ignore_sim=_IGNORE_SIM):
        """
        Adjust the default configuration parameters
        """
        self.min_samples = min_samples
        self.eps         = eps
        self.intp_dist   = intp_dist
        self.gap_penalty = gap_penalty
        self.ignore_dist = ignore_dist
        self.ignore_sim  = ignore_sim

    def _apply_cluster_algo(self):
        """
        Apply the clustering algorithm
        """
        self._compare_all_tracks()
        labels, n_clusters, cluster_ids = self._train_predict_DBSCAN()

        return labels, n_clusters, cluster_ids

    def compare_two_tracks(self, id_trackA, id_trackB):
        """
        Calculate similarity between two tracks by extecuting
        the adapted Needleman-Wunsch algorithm
        """
        gpx1_points = self.tracks_gpx[id_trackA]
        gpx2_points = self.tracks_gpx[id_trackB]

        # Interpolate both tracks to a given step size
        gpx1_points = gps_track_clustering.geo.interpolate_distance(gpx1_points, self.intp_dist)
        gpx2_points = gps_track_clustering.geo.interpolate_distance(gpx2_points, self.intp_dist)

        # Execute adapted Needleman-Wunsch algorithm
        a1, a2 = gps_track_clustering.geo.align_two_tracks(gpx1_points, gpx2_points, self.gap_penalty)

        # Calculate the similarity from the aligned tracks
        match = 0
        for i in range(0, len(a1)):
            if a1[i] is not None and a2[i] is not None:
                match += 1
        total_similar = match / len(a1)

        return a1, a2, total_similar

    def get_tracks_compared(self, id_trackA, id_trackB):
        """
        Compare two tracks and return a geojson FeatureCollection
        """

        # Compare tracks
        a1, a2, total_similar = self.compare_two_tracks(id_trackA, id_trackB)

        # Create geojson with color encoding the differences
        features = []
        for i in range(0, len(a1)):
            if a1[i] is not None and a2[i] is not None:
                thisFeature = Feature(geometry=Point((a1[i].longitude, a1[i].latitude)),
                                      properties={'color': '#378b29'})
                features.append(thisFeature)
                thisFeature = Feature(geometry=Point((a2[i].longitude, a2[i].latitude)),
                                      properties={'color': '#74d680'})
                features.append(thisFeature)
            elif a1[i] is not None and a2[i] is None:
                thisFeature = Feature(geometry=Point((a1[i].longitude, a1[i].latitude)),
                                      properties={'color': '#ff0000'})
                features.append(thisFeature)
            elif a1[i] is None and a2[i] is not None:
                thisFeature = Feature(geometry=Point((a2[i].longitude, a2[i].latitude)),
                                      properties={'color': '#ff7878'})
                features.append(thisFeature)
        feature_collection = FeatureCollection(features)

        return feature_collection, total_similar

    def _compare_all_tracks(self):
        """
        Iterate through all track combinations and calculate similarity
        """

        all_ids = self.summary.id.values.tolist()

        # Load cached distance matrix
        dict_simMatrix_file = self._load_simMatrix()
        print('before {}'.format(len(dict_simMatrix_file)))

        # Calculate all combinations
        n_ids = len(all_ids)
        fact = math.factorial
        n_combinations = int(fact(n_ids) / (fact(n_ids - 2) * fact(2)))

        # Iterate through all track combinations
        n_skipped = 0
        new_data_added = False
        dict_simMatrix = {}
        for i, (idA, idB) in enumerate(itertools.combinations(all_ids, 2)):

            # Regularly save to cache
            if ((i + 1) % 100 == 0) & new_data_added:
                self._write_simMatrix(dict_simMatrix_file)
                new_data_added = False

            # Check if combination was already computed
            identifier = (idA, idB, self.ignore_dist, self.intp_dist, self.gap_penalty)
            if identifier in dict_simMatrix_file:
                n_skipped = n_skipped + 1
                dict_simMatrix[identifier] = dict_simMatrix_file[identifier]
                continue

            # Check distance between start locations
            distance_AB = self._get_distance_start_AB(self, idA, idB, self.summary)

            # Calculate similarity
            if distance_AB < self.ignore_dist:
                a1, a2, total_similar = self.compare_two_tracks(idA, idB)
                print("Compare {}/{} ({}/{})       -> Track Similarity: {:.2%}".format(idA, idB, i, n_combinations, total_similar))
            else:
                total_similar = None
                print("Compare {}/{} ({}/{})       -> Track Similarity: Too far away, skipped... ".format(idA, idB, i, n_combinations))

            dict_simMatrix_file[identifier] = total_similar
            dict_simMatrix[identifier] = total_similar
            new_data_added = True

        print('after {}'.format(len(dict_simMatrix)))
        df_simMatrix = self._dict_to_simMatrix(dict_simMatrix)

        # Cut off all similarity values below certain threshold
        df_simMatrix = df_simMatrix.where(df_simMatrix.fillna(0) > self.ignore_sim, 0)

        # Save to file
        if new_data_added:
            self._write_simMatrix(dict_simMatrix_file)
        self.df_simMatrix = df_simMatrix

    def _get_distance_start_AB(self, idA, idB, summary):
        """
        Calculate the euclidean distance between the start location of two tracks
        """
        startPosA = summary[summary.id == idA]
        startPosB = summary[summary.id == idB]
        distance_AB = gpxpy.geo.distance(startPosA.startpos_lat.values, startPosA.startpos_lon.values, 0,
                                         startPosB.startpos_lat.values, startPosB.startpos_lon.values, 0)
        return distance_AB

    def _write_simMatrix(self, dict_simMatrix):
        """
        Save the similarity matrix to pickle
        """
        if not self.dir_cache:
            return

        fn_out = '{}/simMatrix_{}_{}_{}.pkl'.format(self.dir_cache, self.ignore_dist, self.intp_dist, self.gap_penalty)
        file = open(fn_out, 'wb')
        pickle.dump(dict_simMatrix, file)
        file.close()
        print('_write_simMatrix()')

    def _load_simMatrix(self):
        """
        Load the similarity matrix from pickle
        """
        fn_in = '{}/simMatrix_{}_{}_{}.pkl'.format(self.dir_cache, self.ignore_dist, self.intp_dist, self.gap_penalty)
        try:
            file = open(fn_in, 'rb')
            dict_simMatrix = pickle.load(file)
        except FileNotFoundError:
            dict_simMatrix = {}
        except Exception as e:
            print(e)
            raise
        return dict_simMatrix

    def _dict_to_simMatrix(self, dict_simMatrix):
        """
        Translate a dict containing similarity values to a numpy array
        """
        all_ids = sorted(set([x[0] for x in dict_simMatrix.keys()] + [x[1] for x in dict_simMatrix.keys()]))
        df_simMatrix = pd.DataFrame(columns=all_ids, index=all_ids)

        for (idA, idB, dist_m_ignore, self.intp_dist, self.gap_penalty), sim in dict_simMatrix.items():
            df_simMatrix.at[idA, idB] = sim
            df_simMatrix.at[idB, idA] = sim

        return df_simMatrix

    def _train_predict_DBSCAN(self):
        """
        Execute the DBScan clustering algorithm
        """
        np_simMatrix = self.df_simMatrix.fillna(0).values
        np_simMatrix = 1 - np_simMatrix

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="precomputed")
        db.fit(np_simMatrix)

        # Evaluate results
        # core_samples    = db.core_sample_indices_
        labels          = db.labels_
        n_clusters      = len(set(labels)) - (1 if -1 in labels else 0)
        cluster_ids     = set(labels)

        return labels, n_clusters, cluster_ids
