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
    """ This subclass implements the abstract class Cluster()
        Clustering Method:
            The DBscan algorithm is used with a custom distance metric.
            ...

    Args:
        tracks_gpx (list):       Description of parameter `tracks_gpx`.
        summary (pd.DataFrame):  Each row in the DataFrame represents one activity.
                                 Mandatory columns are: ['id', 'startpos_lat', 'startpos_lon',
                                     'distance'  'duration'  'elevation_gain']
        dir_cache (str):         path to caching dir. None: no caching.
    """

    # Cluster DBSCAN
    _MIN_SAMPLES = 5         # Minimum size of cluster
    _EPS         = 0.05      #

    # Cluster Similarity
    _INTP_DIST   = 30        # [meter], Distance between 2 track points, will be interpolated to this distance
    _GAP_PANALTY = -30       # [meter], Panalty distance between two track points
    _IGNORE_DIST = 500       # [meter], distance between tracks for been not compared

    _IGNORE_SIMI = 0.70      # [percent], Similarity between two tracks below this value will be set to zero

    def __init__(self, tracks_gpx, summary, dir_cache=None):
        Cluster.__init__(self, summary)

        self.tracks_gpx = tracks_gpx
        self.dir_cache = dir_cache

        # CONFIG
        self.desc = 'ClusterTrackSimilarity'
        self.set_config()

        self.df_simMatrix = None

    def set_config(self, min_samples=_MIN_SAMPLES, eps=_EPS, intp_dist=_INTP_DIST,
                   gap_penalty=_GAP_PANALTY, ignore_dist=_IGNORE_DIST):
        self.min_samples = min_samples
        self.eps         = eps
        self.intp_dist   = intp_dist
        self.gap_penalty = gap_penalty
        self.ignore_dist = ignore_dist

    def _apply_cluster_algo(self):

        self._compare_all_tracks()
        labels, n_clusters, cluster_ids = self._train_predict_DBSCAN()

        return labels, n_clusters, cluster_ids

    def compare_two_tracks(self, id_trackA, id_trackB):

        gpx1_points = self.tracks_gpx[id_trackA]
        gpx2_points = self.tracks_gpx[id_trackB]

        # # TODO: Why do I need to recreate the tracks here? They should be there already!!!
        # trackA.set_track_gpx()
        # gpx1_points = trackA.get_track_gpx()
        # trackB.set_track_gpx()
        # gpx2_points = trackB.get_track_gpx()

        gpx1_points = gps_track_clustering.geo.interpolate_distance(gpx1_points, self.intp_dist)
        gpx2_points = gps_track_clustering.geo.interpolate_distance(gpx2_points, self.intp_dist)

        a1, a2 = gps_track_clustering.geo.align_two_tracks(gpx1_points, gpx2_points, self.gap_penalty)

        # Output the difference in the tracks as a percentage
        match = 0
        for i in range(0, len(a1)):
            if a1[i] is not None and a2[i] is not None:
                match += 1
        total_similar = match / len(a1)

        return a1, a2, total_similar

    def get_tracks_compared(self, id_trackA, id_trackB):

        a1, a2, total_similar = self.compare_two_tracks(id_trackA, id_trackB)

        # Create geojson from start locs
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

        ignore_dist = self.ignore_dist
        dict_simMatrix_file = self._load_simMatrix()
        print('before {}'.format(len(dict_simMatrix_file)))

        all_ids = self.summary.id.values.tolist()
        n_ids = len(all_ids)
        fact = math.factorial
        n_combinations = int(fact(n_ids) / (fact(n_ids - 2) * fact(2)))

        n_skipped = 0
        new_data_added = False
        dict_simMatrix = {}
        for i, (idA, idB) in enumerate(itertools.combinations(all_ids, 2)):

            if ((i + 1) % 100 == 0) & new_data_added:
                self._write_simMatrix(dict_simMatrix_file)
                new_data_added = False

            identifier = (idA, idB, ignore_dist, self.intp_dist, self.gap_penalty)
            if identifier in dict_simMatrix_file:
                n_skipped = n_skipped + 1
                dict_simMatrix[identifier] = dict_simMatrix_file[identifier]
                continue

            startPosA = self.summary[self.summary.id == idA]
            startPosB = self.summary[self.summary.id == idB]
            distance_AB = gpxpy.geo.distance(startPosA.startpos_lat.values, startPosA.startpos_lon.values, 0,
                                             startPosB.startpos_lat.values, startPosB.startpos_lon.values, 0)

            if distance_AB < ignore_dist:
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

        df_simMatrix = df_simMatrix.where(df_simMatrix.fillna(0) > 0.70, 0)

        if new_data_added:
            self._write_simMatrix(dict_simMatrix_file)
        self.df_simMatrix = df_simMatrix

    def _write_simMatrix(self, dict_simMatrix):
        if not self.dir_cache:
            return

        fn_out = '{}/simMatrix_{}_{}_{}.pkl'.format(self.dir_cache, self.ignore_dist, self.intp_dist, self.gap_penalty)
        file = open(fn_out, 'wb')
        pickle.dump(dict_simMatrix, file)
        file.close()
        print('_write_simMatrix()')

    def _load_simMatrix(self):
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

        all_ids = sorted(set([x[0] for x in dict_simMatrix.keys()] + [x[1] for x in dict_simMatrix.keys()]))
        df_simMatrix = pd.DataFrame(columns=all_ids, index=all_ids)

        for (idA, idB, dist_m_ignore, self.intp_dist, self.gap_penalty), sim in dict_simMatrix.items():
            df_simMatrix.at[idA, idB] = sim
            df_simMatrix.at[idB, idA] = sim

        return df_simMatrix

    def _train_predict_DBSCAN(self):
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
