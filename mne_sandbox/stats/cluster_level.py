from mne.viz import plot_topomap
from mne.channels import find_ch_adjacency


def _gen_default_cluster_forming_threshold_t(*, n_samples, tail):
    from scipy.stats import t

    p_thresh = 0.05 / (1 + (tail == 0))
    threshold = -t.ppf(p_thresh, n_samples - 1)
    if np.sign(tail) < 0:
        threshold = -threshold
    logger.info("Using a threshold of {:.6f}".format(threshold))
    return threshold


class ClusterTestResult:
    def __init__(
        self,
        *,
        T_values,
        clusters,
        cluster_p_values,
        n_permutations,
        cluster_forming_threshold,
        test_kind,
        n_observations,
        times,
        ch_type,
        info,
        tail,
    ):
        """Container for cluster permutation test results.

        .. note::
           This class is typically instantiated through
           :func:`group_level_cluster_test`. Do not instantiate directly.

        Parameters
        ----------
        T_values : ndarray, shape (n_channels, n_times)
            The T-values of each spatio-temporal node.
        clusters : _type_
            ...
        cluster_p_values : _type_
            ...
        n_permutations : int
            ...
        cluster_forming_threshold : float
            ...
        test_kind : 't-test' | 'F-test'
            ...
        n_observations : int
            ...
        times : ndarray, shape (n_times,)
            The time points.
        ch_type : ...
            ...
        info : Info
            ...
        tail : 'left' | 'right' | 'both'
            ...
        """
        self.T_values = T_values
        self.clusters = clusters
        self.cluster_p_values = cluster_p_values
        self.times = times
        self.info = info
        self.tail = tail
        self.n_permutations = n_permutations
        self.test_kind = test_kind
        self.n_observations = n_observations
        self.cluster_forming_threshold = cluster_forming_threshold
        self.ch_type = ch_type

    def __repr__(self):
        s = (
            f'<ClusterTestResult |\n'
            f' n_clusters={len(self.clusters)},\n'
            f' ch_type="{self.ch_type}",\n'
            f' n_observations={self.n_observations},\n'
            f' n_permutations={self.n_permutations},\n'
            f' cluster_forming_threshold='
            f'{self.cluster_forming_threshold:.6f},\n'
            f' test_kind="{self.test_kind}",\n'
            f' tail="{self.tail}"\n'
            f'>'
        )
        return s

    def _repr_html_(self):
        from ..html_templates import repr_templates_env

        t = repr_templates_env.get_template('clustertestresult.html.jinja')
        t = t.render(
            n_clusters=len(self.clusters),
            ch_type=self.ch_type,
            n_observations=self.n_observations,
            n_permutations=self.n_permutations,
            cluster_forming_threshold=self.cluster_forming_threshold,
            test_kind=self.test_kind,
            tail=self.tail,
        )
        return t

    def to_data_frame(
        self,
        *,
        p_thresh=1,
    ):
        """Convert to a Pandas DataFrame.

        Parameters
        ----------
        p_thresh : float
            Only add clusters below this threshold to the data frame.

        Returns
        -------
        df : DataFrame
            The dataframe.
        """
        import pandas as pd

        columns = [
            'ch_type', 't_start', 't_stop', 'ch_names',
            'cluster_p_value',
            'n_permutations', 'cluster_forming_threshold', 'tail',
            'test_kind', 'n_observations'
        ]
        index = pd.RangeIndex(len(self.clusters), name='cluster_idx')

        df = pd.DataFrame(
            columns=columns,
            index=index,
        )
        for cluster_idx, cluster in enumerate(self.clusters):
            # Cluster time range
            unique_time_indices = sorted(
                set(cluster[0])
            )
            t_start = self.times[unique_time_indices[0]]
            t_stop = self.times[unique_time_indices[-1]]

            # Cluster channel names
            unique_channel_indices = cluster[1]
            ch_names = sorted(
                set([self.info['ch_names'][i] for i in unique_channel_indices])
            )

            # Cluster p-value
            p_val = self.cluster_p_values[cluster_idx]

            df.loc[cluster_idx, 't_start'] = t_start
            df.loc[cluster_idx, 't_stop'] = t_stop
            df.loc[cluster_idx, 'ch_names'] = ', '.join(ch_names)
            df.loc[cluster_idx, 'cluster_p_value'] = p_val

        df['ch_type'] = self.ch_type
        df['n_permutations'] = self.n_permutations
        df['tail'] = self.tail
        df['test_kind'] = self.test_kind
        df['n_observations'] = self.n_observations
        df['cluster_forming_threshold'] =  self.cluster_forming_threshold

        # filter clusters below the set cluster selection threshold
        df = df[df['cluster_p_value'] < p_thresh]

        return df

    def plot_stats(
        self,
        *,
        p_thresh=1,
    ):
        """_summary_

        Parameters
        ----------
        p_thresh : int, optional
            _description_, by default 1

        Returns
        -------
        figs
            A list of figures. The first element is a heatmap of T-values;
            the second contains to topographic plots of the T-values for
            significant clusters, if any.
        """
        import matplotlib.pyplot as plt

        # mask T-values outside of significant clusters with nans
        T_values_sig_clusters = np.nan * np.ones_like(self.T_values)
        for cluster, p_val in zip(self.clusters, self.cluster_p_values):
            if p_val <= p_thresh:
                T_values_sig_clusters[cluster] = self.T_values[cluster]
        del cluster, p_val

        # Todo:
        # [x] vertical line indicating time point zero
        # [x] add information on thresholds, n_permutations etc to results
        #     class
        # [x] add pandas dataframe export
        # [x] plot topos for all significant clusters into a single figure
        # [x] if adjacency is None, call find_adjacency for the specified
        #     ch_type
        # [x] check that all evokeds have the same ch_names, times, etc.
        # [x] sub-set Evokeds for specified ch_type
        # [x] demand ch_type parameter if multiple ch_types are present in the
        #     data
        # [x] add colorbar to topoplots
        # [x] add repr
        # [x] add HTML repr
        # [ ] create lineplot for single-channel data, with horizontal line
        #     for cluster forming threshold
        # [ ] add picks parameter to plot_stats(); this should create 1
        #     line plot per matching channel. For >5, only plot the first 5.
        # [x] the gray cmap is not correct, as it is monotonically
        #     in-/decreasing. Instead, we need a diverging colormap for the
        #     tail='both' case, and need to ensure that higher T-values ->
        #     darker colors for the other cases. For 'both', we need to
        #     merge two existing colormaps.

        if self.tail == 'left':
            vmin = self.T_values.min()
            vmax = 0
            cmap = 'Blues_r'
            cmap_gray = 'Greys_r'
        elif self.tail == 'right':
            vmin = 0
            vmax = self.T_values.max()
            cmap = 'Reds'
            cmap_gray = 'Greys'
        else:  # 'both'
            vmax = np.abs(self.T_values).max()
            vmin = -vmax
            cmap = 'RdBu_r'

            # For gray, we need to combine two existing colormaps, as there is
            # no diverging colormap with gray/black at both endpoints.
            from matplotlib.cm import gray, gray_r
            from matplotlib.colors import ListedColormap

            black_to_white = gray(
                np.linspace(start=0, stop=1, endpoint=False, num=128)
            )
            white_to_black = gray_r(
                np.linspace(start=0, stop=1, endpoint=False, num=128)
            )
            black_to_white_to_black = np.vstack(
                (black_to_white, white_to_black)
            )
            diverging_gray_cmap = ListedColormap(
                black_to_white_to_black, name='DivergingGray'
            )
            cmap_gray = diverging_gray_cmap

        figs = []
        fig, ax = plt.subplots()
        extent = (
            self.times[0],   # left
            self.times[-1],  # right
            len(self.info['ch_names']) - 1,  # bottom
            0  # top
        )

        t_gray = self.T_values.T.copy()
        ax.imshow(
            t_gray, extent=extent, aspect='auto',
            cmap=cmap_gray, vmin=vmin, vmax=vmax, interpolation='none',
        )
        # We store the image object in a variable so we can later use it as a
        # mappable to create colorbars. Note that we do not have to use an
        # image that contains ALL t-values (we're only plotting the significant
        # values here); we simply need an object that contains the correct
        # mapping, which is ensured thanks to the fact that we pass the pre-
        # calculated vmin and vmax here together with the correct cmap.
        t_value_image = ax.imshow(
            T_values_sig_clusters.T, extent=extent, aspect='auto',
            cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none',
        )
        # Add vertical line at time point zero
        if any(self.times < 0) and any(self.times > 0):
            ax.axvline(0, color='black', linestyle='--', lw=0.75)

        # add colorbar
        plt.colorbar(
            ax=ax, shrink=0.75, orientation='vertical', mappable=t_value_image,
            label='T-value'
        )
        # Axis labels and title
        ax.set_title(
            f'T-values thresholded at $p={p_thresh}$'
        )
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channel index')
        figs.append(fig)

        # Find significant clusters.
        significant_clusters_idx = np.where(
            self.cluster_p_values < p_thresh
        )[0]

        if significant_clusters_idx.size == 0:
            logger.info(
                'No significant clusters in the data; not plotting '
                'topographies.'
            )
            return figs
        elif significant_clusters_idx.size > 5:
            logger.warn(
                'More than 5 significant clusters found; plotting '
                'topographies for the first 5.'
            )
            significant_clusters_idx = significant_clusters_idx[:5]

        # Plot topographies for significant clusters
        fig = plt.figure(constrained_layout=False)
        gs = fig.add_gridspec(
            ncols=len(significant_clusters_idx),
            nrows=2,  # 1 for topomaps, 1 for colorbar
            height_ratios=[10, 1],
            hspace=0.05,
            wspace=0.1,
        )

        fig.suptitle(
            'T-value topographies for significant clusters',
            fontweight='bold'
        )

        for plot_number, cluster_idx in enumerate(significant_clusters_idx):
            cluster = self.clusters[cluster_idx]

            time_inds, space_inds = np.squeeze(cluster)
            ch_inds = np.unique(space_inds)
            time_inds = np.unique(time_inds)

            # create spatial mask
            mask = np.zeros((len(self.info['ch_names']), 1), dtype=bool)
            mask[ch_inds, :] = True

            # extract data from significant times and avg over time
            plot_data = self.T_values[time_inds, :].mean(axis=0)

            ax = fig.add_subplot(gs[0, plot_number])
            ax.set_title(
                f'Cluster #{plot_number + 1}\n'
                f'{round(self.times[time_inds[0]], 3):.3f} – '
                f'{round(self.times[time_inds[-1]], 3):.3f} sec'
            )
            plot_topomap(
                data=plot_data,
                pos=self.info,
                axes=ax,
                mask=mask,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                show=False,
            )

        # add colorbar
        colorbar_ax = fig.add_subplot(gs[1, :])
        plt.colorbar(
            cax=colorbar_ax, orientation='horizontal', mappable=t_value_image,
            label='T-value'
        )
        plt.subplots_adjust(top=0.85)
        figs.append(fig)

        return figs


_tail_str_to_int_map = {
    'left': -1,
    'right': 1,
    'both': 0,
}


@verbose
def group_level_cluster_test(
    data,
    *,
    ch_type=None,
    cluster_forming_threshold=None,
    n_permutations=5000,
    tail='both',
    adjacency=None,
    seed=None,
    n_jobs=None,
    verbose=None
):
    """Non-parametric cluster-level test for spatio-temporal(-spectral) data

    Parameters
    ----------
    data : dict
        keys are condition names, values are :class:`~mne.Evoked` or
        :class:`~mne.time_frequency.AverageTFR`
    ch_type : str
        ...
    adjacency : scipy.sparse.spmatrix | None
        ... If ``None``, tries to determine the channel adjacency from the
        data.
    cluster_forming_threshold : float | None
        ...
    %(n_permutations_clust_all)s
    tail : 'left' | 'right' | 'both'
        Whether to perform a one-sided test on the left or right tail, or a
        two sided test. If ``'both'`` (default), a two sided test is performed.
    %(seed)s
    %(n_jobs)s
    %(verbose)s
    """
    # XXX add support for AverageTFR
    # XXX check for consistent input size across conditions
    from .. import combine_evoked  # avoid circular import

    _validate_type(data, types=dict, item_name='data')
    if len(data) == 0 or len(data) > 2:
        raise ValueError('Data must contain one or two elements.'
                         f'Got {len(data)}.')

    if ch_type is None:
        evoked = list(data.values())[0][0]
        evoked.pick_types(meg=True, eeg=True, fnirs=True, ecog=True)
        ch_types = set(evoked.get_channel_types())
        if len(ch_types) > 1:
            raise ValueError(
                'Multiple channel types found in data. Please specify '
                'ch_type.'
            )
        ch_type = list(ch_types)[0]
        del ch_types

    # Pick channel types.
    # XXX This creates an unnecessary copy of the data if no picking is needed.
    for condition, evokeds in data.items():
        data[condition] = [
            e.copy().pick(ch_type)
            for e in evokeds
        ]
    del condition, evokeds

    # Ensure times and channels match across evokeds.
    all_evokeds = []
    for evokeds in data.values():
        all_evokeds.extend(evokeds)

    times = all_evokeds[0].times
    ch_names = all_evokeds[0].ch_names

    for evoked in all_evokeds[1:]:
        if not np.array_equal(evoked.times, times):
            raise ValueError(
                'All evokeds must have the same times.'
            )
        if not np.array_equal(evoked.ch_names, ch_names):
            raise ValueError(
                'All evokeds must have the same ch_names, and in the same '
                'order.'
            )

    del all_evokeds, evokeds, times, ch_names

    # If two conditions were provided, calculate the difference – we will
    # later perform a one-sample t-test against zero.
    if len(data) == 2:
        evoked_diff = []
        for evoked1, evoked2 in zip(*data.values()):
            diff = combine_evoked(
                [evoked1, evoked2],
                weights=[1, -1]
            )
            evoked_diff.append(diff)

        data = {'diff': evoked_diff}
        del diff, evoked_diff, evoked1, evoked2


    # data now is a dictionary with only a single key
    # We can now extract the evoked data as a NumPy array.
    data_array = [e.data for e in list(data.values())[0]]
    data_array = np.asarray(data_array)

    # spatio_temporal_cluster_1samp_test expects spatial dimension last
    # expected dimensions: observations (difference) x time
    # (x frequency) x sensors / vertices
    data_array = np.transpose(data_array, [0, 2, 1])

    if adjacency is None:
        adjacency, _ = find_ch_adjacency(
            info=list(data.values())[0][0].info,
            ch_type=ch_type
        )

    # now feed the data to the actual stats function
    result = spatio_temporal_cluster_1samp_test(
        data_array,
        threshold=cluster_forming_threshold,
        n_permutations=n_permutations,
        tail=_tail_str_to_int_map[tail],
        adjacency=adjacency,
        seed=seed,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    T_values, clusters, cluster_p_values, _ = result
    del result

    if len(data) <= 2:
        test_kind = 't-test'
    else:
        raise NotImplementedError('F-test not implemented yet.')

    # If clussuter_forming_threshold was not passed, calculate it the same
    # way that spatio_temporal_cluster_1samp_test() does, so we can store it
    # in the result.
    if cluster_forming_threshold is None:
        cluster_forming_threshold = _gen_default_cluster_forming_threshold_t(
            n_samples=len(data_array),
            tail=_tail_str_to_int_map[tail],
        )

    result = ClusterTestResult(
        T_values=T_values,
        clusters=clusters,
        cluster_p_values=cluster_p_values,
        times=list(data.values())[0][0].times,
        info=list(data.values())[0][0].info,
        tail=tail,
        cluster_forming_threshold=cluster_forming_threshold,
        n_permutations=n_permutations,
        n_observations=len(data_array),
        test_kind=test_kind,
        ch_type=ch_type,
    )

    return result
