import logging

import numpy as np

from . import utils


class MsrConnectionManager:
    """Tracks connection matrices for various models, enabling efficient caching.

    This class maintains various connection matrices used in the multiscale simulation, improving efficiency by caching them when necessary.

    Args:
        config (MsrConfig): The multiscale run configuration.
    """

    def __init__(self, config, managers):
        """Initialize the MsrConnectionManager with a given configuration.

        Args:
            config (MsrConfig): The multiscale run configuration.
            managers (dict): Managers dictionary.
        """
        self.config = config
        self.managers = managers
        # needed for the merge syncing scheme
        self.cache = {}
        self.pyeval = utils.PyExprEval(self.config)

    @utils.cache_decorator(
        only_rank0=False,
        field_names=["nXsecMat", "nsecXnsegMat", "nXnsegMatBool"],
    )
    @utils.logs_decorator
    def connect_neurodamus2neurodamus(self):
        """Add some useful matrices that map ndam points, segments, sections, and neurons.

        This method calculates and stores several matrices that provide mappings between various components of neuronal and segment data.

        Returns
            None
        """
        ndam_m = self.managers["neurodamus"]
        pts = ndam_m.get_seg_points(self.config.multiscale_run.mesh_scale)

        self.nXsecMat = ndam_m.get_nXsecMat()
        self.nsecXnsegMat = ndam_m.get_nsecXnsegMat(pts)
        self.nXnsegMatBool = self.nXsecMat.dot(self.nsecXnsegMat) > 0

    @utils.cache_decorator(
        only_rank0=False,
        field_names=["nsegXtetMat", "nXtetMat"],
    )
    @utils.logs_decorator
    def connect_neurodamus2steps(self):
        """Neuron volume fractions in tets.

        This method calculates the neuron volume fractions within tetrahedral elements and stores the results in connection matrices.

        Returns
            None
        """
        ndam_m = self.managers["neurodamus"]
        steps_m = self.managers["steps"]

        pts = ndam_m.get_seg_points(scale=self.config.multiscale_run.mesh_scale)

        # Connection matrices
        self.nsegXtetMat = steps_m.get_nsegXtetMat(pts)
        self.nXtetMat = self.nXsecMat.dot(self.nsecXnsegMat.dot(self.nsegXtetMat))

    @utils.cache_decorator(
        only_rank0=True,
        field_names=["tetXbfVolsMat", "tetXbfFlowsMat", "tetXtetMat"],
    )
    @utils.logs_decorator
    def connect_bloodflow2steps(self):
        """Volumes and flows fractions in tets.

        This method calculates and stores volumes and flows fractions within tetrahedral elements and synchronization matrices.

        Returns
            None
        """
        bf_m = self.managers["bloodflow"]
        steps_m = self.managers["steps"]

        pts = None
        if utils.rank0():
            pts = bf_m.get_seg_points(steps_m.msh._scale)
        pts = utils.comm().bcast(pts, root=0)

        mat, starting_tets = steps_m.get_tetXbfSegMat(pts)
        self.tetXbfVolsMat, self.tetXbfFlowsMat = None, None
        if utils.rank0():
            # resize based on tet volume
            flowsMat = mat.sign()

            for isec, itet in starting_tets:
                if isec not in bf_m.entry_nodes:
                    flowsMat[itet, isec] = 0

            flowsMat.eliminate_zeros()

            # Connection matrices
            self.tetXtetMat = steps_m.get_tetXtetMat()
            self.tetXbfVolsMat = mat
            self.tetXbfFlowsMat = flowsMat

    def _apply_custom_op(self, con, vals) -> np.ndarray:
        """Evaluate custom op conversion.

        Args:
            con (MsrConfig): Connection object from the MsrConfig object.
            vals: values to be changed by the custom op operation.

        Returns:
            The changed values.
        """
        if "transform_expression" in con:
            vals = self.pyeval(
                con.transform_expression,
                vals=vals,
                nXsecMat=getattr(self, "nXsecMat", None),
                nsecXnsegMat=getattr(self, "nsecXnsegMat", None),
                nXnsegMatBool=getattr(self, "nXnsegMatBool", None),
                nsegXtetMat=getattr(self, "nsegXtetMat", None),
                tetXbfVolsMat=getattr(self, "tetXbfVolsMat", None),
                tetXbfFlowsMat=getattr(self, "tetXbfFlowsMat", None),
                tetXtetMat=getattr(self, "tetXtetMat", None),
            )

        return vals

    def _set_vals(self, con, vals: np.ndarray, action: str) -> None:
        """Set values in the destination simulator

        It also sets the values in the source simulator if it is not a demoted-to-set action.

        Args:
            con (MsrConfig): Connection object from the MsrConfig object.
            vals: values to be set.
            action: Syncing scheme. Details in: process_syncs. Not all the actions need to be
                implemented for all the sync types.
        """

        dest_simulator = self.managers[con.dest_simulator]
        getattr(dest_simulator, con.dest_set_func)(vals=vals, **con.dest_set_kwargs)
        logging.debug(f"vals set in dest: {con.dest_simulator}")
        # if specified, set values to the source simulator too
        if "src_set_func" in con and "src_set_kwargs" in con and con.action == action:
            src_simulator = self.managers[con.src_simulator]
            # Set merged vals into srcs
            getattr(src_simulator, con.src_set_func)(vals=vals, **con.src_set_kwargs)
            logging.debug(f"vals set in src: {con.src_simulator}")

    @utils.logs_decorator
    def neurodamus2steps_sync(
        self, sync_event: str, icon: int, con, action: str
    ) -> None:
        """Syncs data from 'MsrNeurodamusManager' (source) to 'MsrStepsManager' (destination).

        Args:
            sync_event: event tag. It specifies when this sync takes place in the main loop.
            icon: connection index. Position in the array of connections. It is important because later
                connections override previous ones.
            con (MsrConfig): Connection object from the MsrConfig object.
            action: Syncing scheme. Details in: process_syncs. Not all the actions need to be
                implemented for all the sync types.

        Returns:
            None

        Raises:
            NotImplementedError: If the action is not implemented.
        """

        match action:
            case "sum":
                vals = getattr(self.managers[con.src_simulator], con.src_get_func)(
                    **con.src_get_kwargs
                )
                # apply standard transformations
                vals = self.nsegXtetMat.transpose().dot(vals)
                # apply custom transformations
                vals = self._apply_custom_op(con=con, vals=vals)
                # distribute the results
                utils.comm().Allreduce(vals, vals, op=utils.mpi().SUM)
                # set the vals
                self._set_vals(con=con, vals=vals, action=action)

            case _:
                raise NotImplementedError(
                    f"action: '{action}' not implemented! Available actions: ['sum']"
                )

        if con.action == "merge":
            self.cache[(sync_event, icon)] = np.copy(vals)

    @utils.logs_decorator
    def neurodamus2metabolism_sync(
        self, sync_event: str, icon: int, con, action: str
    ) -> None:
        """Syncs data from 'MsrNeurodamusManager' (source) to 'MsrMetabolismManager' (destination).

        Args:
            sync_event: event tag. It specifies when this sync takes place in the main loop.
            icon: connection index. Position in the array of connections. It is important because later
                connections override previous ones.
            con (MsrConfig): Connection object from the MsrConfig object.
            action: Syncing scheme. Details in: process_syncs. Not all the actions need to be
                implemented for all the sync types.

        Returns:
            None

        Raises:
            NotImplementedError: If the action is not implemented.
        """
        ndam_m = self.managers["neurodamus"]

        match action:
            case "set":
                vals = getattr(self.managers[con.src_simulator], con.src_get_func)(
                    **con.src_get_kwargs
                )
                # apply standard transformations
                vals = self.nXnsegMatBool.dot(vals)
                vals = np.divide(vals, ndam_m.nc_weights[con.src_get_kwargs.weight][0])
                # apply custom transformations
                vals = self._apply_custom_op(con=con, vals=vals)
                # set the vals
                self._set_vals(con=con, vals=vals, action=action)

            case "merge":
                # Merging means doing this operation:
                # delta Vals_src + delta Vals_dest - Vals_previous sync = new Vals

                # If Vals_previous is missing, downgrade to setting
                if (sync_event, icon) not in self.cache:
                    self.neurodamus2metabolism_sync(
                        sync_event=sync_event, icon=icon, con=con, action="set"
                    )
                    return

                # Collect and adapt vals from src
                vals = getattr(self.managers[con.src_simulator], con.src_get_func)(
                    **con.src_get_kwargs
                )
                # apply standard transformations
                vals = self.nXnsegMatBool.dot(vals)
                vals = np.divide(vals, ndam_m.nc_weights[con.src_get_kwargs.weight][0])
                # Add vals from dest
                dest_vals = getattr(
                    self.managers[con.dest_simulator], con.dest_get_func
                )(**con.dest_get_kwargs)
                # remove Vals_previous
                prev_vals = self.cache[(sync_event, icon)]
                # merge
                vals += dest_vals - prev_vals
                # apply additional operations
                vals = self._apply_custom_op(con=con, vals=vals)
                # set values
                self._set_vals(con=con, vals=vals, action=action)

            case _:
                raise NotImplementedError(
                    f"action: '{action}' not implemented! Available actions: ['set', 'merge']"
                )

        if con.action == "merge":
            self.cache[(sync_event, icon)] = np.copy(vals)

    def neurodamus2bloodflow_sync(
        self, sync_event: str, icon: int, con, action: str
    ) -> None:
        """Syncs data from 'MsrNeurodamusManager' (source) to 'MsrBloodflowManager' (destination).

        Args:
            sync_event: event tag. It specifies when this sync takes place in the main loop.
            icon: connection index. Position in the array of connections. It is important because later
                connections override previous ones.
            con (MsrConfig): Connection object from the MsrConfig object.
            action: Syncing scheme. Details in: process_syncs. Not all the actions need to be
                implemented for all the sync types.

        Returns:
            None

        Raises:
            NotImplementedError: If the action is not implemented.
        """

        match action:
            case "set":
                # Probably the radii
                # idxs of the vasculature segments
                idxs, vals = getattr(
                    self.managers[con.src_simulator], con.src_get_func
                )(**con.src_get_kwargs)
                idxs = utils.comm().gather(idxs, root=0)
                vals = utils.comm().gather(vals, root=0)
                if utils.rank0():
                    idxs = [j for i in idxs for j in i]
                    vals = [j for i in vals for j in i]
                    # apply additional custom operations
                    vals = self._apply_custom_op(con=con, vals=vals)
                    # this is custom made because here we need to service also the indexes
                    getattr(self.managers[con.dest_simulator], con.dest_set_func)(
                        idxs=idxs, vals=vals, **con.dest_set_kwargs
                    )
                    logging.info(f"vals set in dest: {con.dest_simulator}")
            case _:
                raise NotImplementedError(
                    f"action: '{action}' not implemented! Available actions: ['set']"
                )

        if con.action == "merge":
            self.cache[(sync_event, icon)] = np.copy(vals)

    def bloodflow2metabolism_sync(
        self, sync_event: str, icon: int, con, action: str
    ) -> None:
        """Syncs data from 'MsrBloodflowManager' (source) to 'MsrMetabolismManager' (destination).

        Args:
            sync_event: event tag. It specifies when this sync takes place in the main loop.
            icon: connection index. Position in the array of connections. It is important because later
                connections override previous ones.
            con (MsrConfig): Connection object from the MsrConfig object.
            action: Syncing scheme. Details in: process_syncs. Not all the actions need to be
                implemented for all the sync types.

        Returns:
            None

        Raises:
            NotImplementedError: If the action is not implemented.
        """

        match action:
            case "set":
                vals = None
                if utils.rank0():
                    # get raw values
                    vals = getattr(self.managers[con.src_simulator], con.src_get_func)(
                        **con.src_get_kwargs
                    )

                    # apply additional operations

                    # In this operation there should be this multiplier: 1e-12 * 500 = 5e-10

                    # 1e-12 to pass from um^3 to ml and from um^3/s to ml/s.
                    # 500 is 1/0.0002 (1/0.2%) since we discussed that the vasculature is only 0.2% of the total
                    # and it is not clear to what the winter paper is referring to exactly for volume and flow
                    # given that we are sure that we are not double counting on a tet Fin and Fout, we can use
                    # and abs value to have always positive input flow

                    vals = self._apply_custom_op(con=con, vals=vals)

                vals = utils.comm().bcast(vals, root=0)

                vals = self.nXtetMat.dot(vals)

                self._set_vals(con=con, vals=vals, action=action)

            case _:
                raise NotImplementedError(
                    f"action: '{action}' not implemented! Available actions: ['set']"
                )

        if con.action == "merge":
            self.cache[(sync_event, icon)] = np.copy(vals)

    @utils.logs_decorator
    def steps2metabolism_sync(
        self, sync_event: str, icon: int, con, action: str
    ) -> None:
        """Syncs data from 'MsrStepsManager' (source) to 'MsrMetabolismManager' (destination).

        Args:
            sync_event: event tag. It specifies when this sync takes place in the main loop.
            icon: connection index. Position in the array of connections. It is important because later
                connections override previous ones.
            con (MsrConfig): Connection object from the MsrConfig object.
            action: Syncing scheme. Details in: process_syncs. Not all the actions need to be
                implemented for all the sync types.

        Returns:
            None

        Raises:
            NotImplementedError: If the action is not implemented.
        """

        match action:
            case "set":
                # get the raw vals
                vals = getattr(self.managers[con.src_simulator], con.src_get_func)(
                    **con.src_get_kwargs
                )
                # custom operations
                vals = self._apply_custom_op(con=con, vals=vals)
                # standard transformations
                vals = self.nXtetMat.dot(vals)
                # set vals
                self._set_vals(con=con, vals=vals, action=action)

            case _:
                raise NotImplementedError(
                    f"action: '{action}' not implemented! Available actions: ['set']"
                )

        if con.action == "merge":
            self.cache[(sync_event, icon)] = np.copy(vals)

    @utils.logs_decorator
    def process_syncs(self, sync_event: str) -> None:
        """Dispatch to the correct syncing procedure based on the `sync_event`

        The sync event marks, in the main loop, when the particular series of connections syncs take place.

        There are 3 ways to sync that may or may not be implemented depending on what was needed:

        - sum: the values from the source are added to the values of destination and inserted in destination
        - set: the values from the source override the values in destination
        - merge: in this case we consider source and destination as having the same level of priority.

        It is appropriate when one simulator produces something and the other one consumes it. In formulae:

        delta_val_src + delta_val_dest + previous_val = val

        Since we usually do not have deltas, we can convert it to:

        val_src + val_dest - previous_val = val

        After, the val is set for both, source and destination. As a side effect, results are
        source <-> desitination swap invaritant (obviously not the case for the other schemes) apart from
        the fact that the sync may be done in different parts of the code.

        Args:
            dest_manager_name: destination manager name.

        Returns:
            None
        """

        logging.info(f"processing sync event: {sync_event}")

        for icon, con in enumerate(self.config.multiscale_run.connections[sync_event]):
            if not self.config.is_manager_active(
                con.src_simulator
            ) or not self.config.is_manager_active(con.dest_simulator):
                continue

            logging.info(f"   connect: {con.src_simulator} -> {con.dest_simulator}")

            getattr(self, f"{con.src_simulator}2{con.dest_simulator}_sync")(
                sync_event=sync_event, icon=icon, con=con, action=con.action
            )

    @utils.logs_decorator
    def remove_gids(self, failed_cells: list[int]):
        """Remove igids from the various connection matrices, metabolism and neurodamus

        Args:
            failed_cells: list of errors for the failed cells. None if the cell is working.

        Returns:
            None
        """
        ndam_m = self.managers["neurodamus"]
        metab_m = self.managers.get("metabolism", None)

        gids = ndam_m.gids
        failed_gids = {
            gids[igid]: e for igid, e in enumerate(failed_cells) if e is not None
        }
        ndam_m.removed_gids |= failed_gids

        to_be_removed = list(ndam_m.gen_to_be_removed_segs())

        if hasattr(self, "nsegXtetMat"):
            self.nsegXtetMat = utils.delete_rows(self.nsegXtetMat, to_be_removed)
        if hasattr(self, "nXnsegMatBool"):
            self.nXnsegMatBool = utils.delete_cols(self.nXnsegMatBool, to_be_removed)

        to_be_removed = list(
            [idx for idx, e in enumerate(failed_cells) if e is not None]
        )
        if hasattr(self, "nXtetMat"):
            self.nXtetMat = utils.delete_rows(self.nXtetMat, to_be_removed)
        if hasattr(self, "nXnsegMatBool"):
            self.nXnsegMatBool = utils.delete_rows(self.nXnsegMatBool, to_be_removed)

        for k, v in self.cache.items():
            self.cache[k] = utils.remove_elems(v=v, to_be_removed=to_be_removed)

        if metab_m is not None:
            metab_m.parameters = utils.delete_rows(metab_m.parameters, to_be_removed)
            metab_m.vm = utils.delete_rows(metab_m.vm, to_be_removed)

        # keep this as last
        ndam_m.ncs = utils.remove_elems(v=ndam_m.ncs, to_be_removed=to_be_removed)
        for k, v in ndam_m.nc_weights.items():
            ndam_m.nc_weights[k] = (
                utils.remove_elems(v=v[0], to_be_removed=to_be_removed),
                utils.remove_elems(v=v[1], to_be_removed=to_be_removed),
            )

        ndam_m.check_neuron_removal_status()
