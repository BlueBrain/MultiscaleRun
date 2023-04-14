from __future__ import print_function

import os
import logging

import numpy as np
import pandas as pd
from diffeqpy import de
from julia import Main

# the object MPI exists already. It is the steps one
from mpi4py import MPI as MPI4PY
from neurodamus import Neurodamus
from neurodamus.core import ProgressBarRank0 as ProgressBar
from neurodamus.utils.logging import log_stage
from neurodamus.utils.timeit import TimerManager, timeit

comm = MPI4PY.COMM_WORLD
import multiscale_run.dualrun.timer.mpi as mt
import multiscale_run.dualrun.sec_mapping.sec_mapping as sec_mapping

# Memory tracking
import psutil

from multiscale_run import (
    utils,
    steps_utils,
    metabolism_utils,
    neurodamus_utils,
    bloodflow_manager,
    printer,
)

import config

logging.basicConfig(level=logging.DEBUG)


def main():
    rank: int = comm.Get_rank()

    config.print_config()
    prnt = printer.MsrPrinter()

    with timeit(name="initialization"):

        ndamus = Neurodamus(
            config.sonata_path,
            logging_level=None,
            enable_coord_mapping=True,
            cleanup_atexit=False,
        )

        # Times are in ms (for NEURON, because STEPS works with SI)
        DT = ndamus._run_conf[
            "Dt"
        ]  # 0.025  #ms i.e. = 25 usec which is timstep of ndam
        SIM_END = ndamus._run_conf["Duration"]  # 500.0 #10.0 #1000.0 #ms
        SIM_END_coupling_interval = DT * config.dt_nrn2dt_jl
        logging.info(f"DT: {DT}")
        logging.info(f"SIM_END: {SIM_END}")
        logging.info(f"SIM_END_coupling_interval: {SIM_END_coupling_interval}")

        # In steps use M/L and apply the SIM_REAL ratio
        CA = (
                config.COULOMB
                / config.AVOGADRO
                * config.CONC_FACTOR
                * (DT * 1e3 * config.dt_nrn2dt_steps)
        )

        logging.info(f"Initializing simulations...")

        ndamus.sim_init()
        rss = []  # Memory tracking
        # TODO explanation of every entry that eventually will appear in this vector?
        um = {}  # u stands for Julia ODE var and m stands for metabolism

        if config.with_steps:
            log_stage("Initializing steps model and mesh...")

            (
                steps_sim,
                neurSecmap,
                ntets,
                global_inds,
                index,
                tetVol,
            ) = steps_utils.init_steps(ndamus)

            tetConcs = np.zeros((ntets,), dtype=float)
            tet_currents_all = np.zeros((ntets,), dtype=float)

        if config.with_metabolism:
            u0 = pd.read_csv(config.u0_file, sep=",", header=None)[0].tolist()

            logging.info(f"get volumes")
            cells_volumes = neurodamus_utils.get_cell_volumes(ndamus)
            gid_to_cell = {
                int(nc.CCell.gid): nc for nc in ndamus.circuits.get_node_manager("All").cells
            }
            all_needs = comm.reduce(
                {rank: set(gid_to_cell.keys())},
                op=utils.join_dict,
                root=0,
            )
            if rank == 0:
                assert len([val for sublist in all_needs.values() for val in sublist]) > 0
                all_needs.pop(0)

            ATDPtot_n = metabolism_utils.load_metabolism_data(Main)
            metabolism = metabolism_utils.gen_metabolism_model(Main)

        if config.with_bloodflow:
            bf_manager = bloodflow_manager.MsrBloodflowManager(ndamus)

            Nmat = neurodamus_utils.get_Nmat(ndamus, ntets, neurSecmap)
            Mmat_flow = bf_manager.get_Mmat_flow(ntets)
            Mmat_vol = bf_manager.get_Mmat_vol(ntets)
            Tmat = steps_utils.get_Tmat(tetVol)

            bf2n_flow = Nmat.dot(Tmat).dot(Mmat_flow)
            bf2n_vol = Nmat.dot(Tmat).dot(Mmat_vol)



    log_stage("===============================================")
    log_stage("Running the selected solvers ...")

    steps, idxm = 0, 0
    for t in ProgressBar(int(SIM_END / (config.dt_nrn2dt_steps * DT)))(
            utils.timesteps(SIM_END, DT * config.dt_nrn2dt_steps)
    ):
        steps += config.dt_nrn2dt_steps
        with timeit(name="main_loop"):
            with timeit(name="neurodamus_solver"):

                ndamus.solve(t)

            if config.with_steps:
                with timeit(name="steps_loop"):
                    log_stage("steps loop")

                    with timeit(name="steps_solver"):
                        steps_sim.run(t / 1000)  # ms to sec

                    with timeit(name="neurodamus_steps_feedback"):
                        tet_currents = sec_mapping.fract_collective(
                            neurSecmap, ntets, global_inds
                        )

                        comm.Allreduce(tet_currents, tet_currents_all, op=MPI4PY.SUM)

                        # update the tet concentrations according to the currents
                        steps_sim.stepsSolver.getBatchTetSpecConcsNP(
                            index, config.Na.name, tetConcs
                        )
                        # 0.001A/mA 6.24e18 particles/coulomb 1000L/m3
                        tetConcs = tetConcs + tet_currents_all * CA * tetVol
                        steps_sim.stepsSolver.setBatchTetSpecConcsNP(
                            index, config.Na.name, tetConcs
                        )

                        prnt.append_to_file(
                            config.moles_current_output,
                            [t, sum(tetConcs * tetVol), sum(tet_currents_all)],
                        )

                        prnt.append_to_file(
                            config.comp_counts_output,
                            [
                                t,
                                *[
                                    steps_sim.stepsSolver.getCompSpecCount(
                                        config.Mesh.compname, spec
                                    )
                                    for spec in config.specNames
                                ],
                            ],
                        )

            if (steps % config.dt_nrn2dt_jl == 0) and config.with_metabolism:
                with timeit(name="metabolism_loop"):
                    log_stage("metabolism loop")
                    outs_r_glu, outs_r_gaba = {}, {}
                    (
                        collected_num_releases_gaba,
                        collected_num_releases_glutamate,
                    ) = neurodamus_utils.collect_gaba_glutamate_releases(ndamus)

                    comm.Barrier()
                    for k, v in neurodamus_utils.release_sums(
                            collected_num_releases_glutamate
                    ).items():
                        prnt.append_to_file(
                            config.ins_glut_file_output,
                            [idxm, t, v],
                            comm.Get_rank(),
                        )
                    comm.Barrier()
                    for k, v in neurodamus_utils.release_sums(
                            collected_num_releases_gaba
                    ).items():
                        prnt.append_to_file(
                            config.ins_gaba_file_output,
                            [idxm, t, v],
                            comm.Get_rank(),
                        )

                    logging.info(f"collect all_events glut")
                    all_outs_r_glut = neurodamus_utils.collect_received_events(
                        collected_num_releases_glutamate,
                        all_needs,
                        gid_to_cell,
                        outs_r_glu,
                    )
                    for sgid, v in all_outs_r_glut.items():
                        prnt.append_to_file(
                            config.outs_glut_file_output, [idxm, sgid, v]
                        )

                    logging.info(f"collect all_events gaba")
                    all_outs_r_gaba = neurodamus_utils.collect_received_events(
                        collected_num_releases_gaba, all_needs, gid_to_cell, outs_r_gaba
                    )
                    for sgid, v in all_outs_r_gaba.items():
                        prnt.append_to_file(
                            config.outs_gaba_file_output, [idxm, sgid, v]
                        )

                    comm.Barrier()


                    with timeit(name="neurodamus_metabolism_feedback"):

                        (
                            ina_density,
                            ina_density_gids_without_valid_segs,
                        ) = neurodamus_utils.get_current_avgs(
                            gid_to_cell=gid_to_cell, seg_filter=config.Na.current_var, weights="area"
                        )
                        (
                            ik_density,
                            ik_density_gids_without_valid_segs,
                        ) = neurodamus_utils.get_current_avgs(
                            gid_to_cell=gid_to_cell,
                            seg_filter=config.KK.current_var, weights="area"
                        )
                        (
                            nais_mean,
                            nais_mean_gids_without_valid_segs,
                        ) = neurodamus_utils.get_current_avgs(
                            gid_to_cell=gid_to_cell,
                            seg_filter=config.Na.nai_var, weights="volume"
                        )
                        (
                            kis_mean,
                            kis_mean_gids_without_valid_segs,
                        ) = neurodamus_utils.get_current_avgs(
                            gid_to_cell=gid_to_cell,
                            seg_filter=config.KK.ki_var, weights="volume"
                        )
                        (
                            cais_mean,
                            cais_mean_gids_without_valid_segs,
                        ) = neurodamus_utils.get_current_avgs(
                            gid_to_cell=gid_to_cell,
                            seg_filter=config.Ca.current_var, weights="volume"
                        )
                        (
                            atpi_mean,
                            atpi_mean_gids_without_valid_segs,
                        ) = neurodamus_utils.get_current_avgs(
                            gid_to_cell=gid_to_cell,
                            seg_filter=config.ATP.atpi_var, weights="volume"
                        )
                        (
                            adpi_mean,
                            adpi_mean_gids_without_valid_segs,
                        ) = neurodamus_utils.get_current_avgs(
                            gid_to_cell=gid_to_cell,
                            seg_filter=config.ADP.adpi_var, weights="volume"
                        )

                        gids_without_valid_segs = (
                                ina_density_gids_without_valid_segs
                                & ik_density_gids_without_valid_segs
                                & nais_mean_gids_without_valid_segs
                                & kis_mean_gids_without_valid_segs
                                & cais_mean_gids_without_valid_segs
                                & atpi_mean_gids_without_valid_segs
                                & adpi_mean_gids_without_valid_segs
                        )

                        for c_gid in gids_without_valid_segs:
                            prnt.append_to_file(
                                config.test_counter_seg_file,
                                c_gid,
                            )
                    comm.Barrier()


                    log_stage(f"prepare metabolism param, len(c_gid) = {len(gid_to_cell)}")
                    print(gid_to_cell)
                    failed_cells = set()
                    for c_gid, nc in neurodamus_utils.gen_ncs(gid_to_cell):
                        print(cells_volumes)

                        logging.info(f"metabolism, processing c_gid: {c_gid}")


                        GLY_a, mito_scale = config.get_GLY_a_and_mito_vol_frac(c_gid)

                        # u0 = [-65.0, config.m0, *u0fromFile]
                        # u0 = [VNeu0,m0,h0,n0,Conc_Cl_out,Conc_Cl_in, Na0in,K0out,Glc_b,Lac_b,O2_b,Q0,Glc_ecs,Lac_ecs,O2_ecs,O2_n,O2_a,Glc_n,Glc_a,Lac_n,Lac_a,Pyr_n,Pyr_a,PCr_n,PCr_a,Cr_n,Cr_a,ATP_n,ATP_a,ADP_n,ADP_a,NADH_n,NADH_a,NAD_n,NAD_a,ksi0,ksi0]

                        tspan_m = (
                            1e-3 * float(idxm) * SIM_END_coupling_interval,
                            1e-3 * (float(idxm) + 1.0) * SIM_END_coupling_interval,
                        )  # tspan_m = (float(t/1000.0),float(t/1000.0)+1) # tspan_m = (float(t/1000.0)-1.0,float(t/1000.0))
                        um[(0, c_gid)] = u0

                        # ini GLY_a c_gid dependent

                        # um[(0, c_gid)][127] = GLY_a
                        vm = um[(idxm, c_gid)]

                        # vm[161] = vm[161] - outs_r_glu.get(c_gid, 0.0)*4000.0/(6e23*1.5e-12)
                        # vm[165] = vm[165] - outs_r_gaba.get(c_gid, 0.0)*4000.0/(6e23*1.5e-12)

                        vm[22] = (
                                0.5 * 1.384727988648391 + 0.5 * atpi_mean[c_gid]
                        )  # atpi_mean[c_gid] #0.5 * 2.2 + 0.5 * atpi_mean[c_gid] # 23 in jl

                        vm[23] = 0.5 * 1.384727988648391 / 2 * (
                                -0.92
                                + np.sqrt(
                            0.92 * 0.92
                            + 4 * 0.92 * (ATDPtot_n / 1.384727988648391 - 1)
                        )
                        ) + 0.5 * atpi_mean[c_gid] / 2 * (
                                         -0.92
                                         + np.sqrt(
                                     0.92 * 0.92
                                     + 4 * 0.92 * (ATDPtot_n / atpi_mean[c_gid] - 1)
                                 )
                                 )

                        vm[98] = nais_mean[
                            c_gid
                        ]  # old idx: 6 # idx 99 in jl, but py is 0-based and jl is 1-based # Na_n

                        # vm[95] = um[(0, c_gid)][95] - 1.33 * ( kis_mean[c_gid] - 140.0 )  # u0[7] - 1.33 * (kis_mean[c_gid] - 140.0) # old idx: 7 # K_out
                        vm[95] = 3.0 - 1.33 * (kis_mean[
                                                   c_gid] - 140.0)  # u0[7] - 1.33 * (kis_mean[c_gid] - 140.0) # old idx: 7 # K_out

                        logging.info(
                            f"------------------------ NDAM FOR METAB: {', '.join(str(i) for i in [vm[22], vm[23], vm[98], vm[95], um[(0, c_gid)][95], kis_mean[c_gid]])}"
                        )

                        # Katta: "Polina suggested the following asserts as rule of thumb. In this way we detect
                        # macro-problems like K+ accumulation faster. For now the additional computation is minimal.
                        # Improvements are possible if needed."
                        assert 0.25 <= vm[22] <= 2.5 # assert 0.7 <= vm[95] <= 2.3 # TODO make it stricter once more things are in place
                        assert 7 <= vm[98] <= 30 # usually around 10 # TODO recheck this once everything is in place
                        assert 2.5 <= vm[95] <= 20  # assert 2.5 <= vm[95] <= 8 # TODO switch once K+ accumulation is fully fixed
                        assert 2.5 <= um[(0, c_gid)][95] <= 20 # assert 2.5 <= um[(0, c_gid)][95] <= 8 # TODO switch once K+ accumulation is fully fixed
                        assert 120 <= kis_mean[c_gid] <= 160

                        # 2.2 should coincide with the BC METypePath field & with u0_file
                        # commented on 13jan2021 because ATPase is in model, so if uncomment, the ATPase effects will be counted twice for metab model
                        # commented on 13jan2021 because ATPase is in model, so if uncomment, the ATPase effects will be counted twice for metab model

                        # TODO : Here goes the coupling with Blood flow solver

                        param = [
                            ina_density[c_gid],
                            ik_density[c_gid],
                            mito_scale,
                        ]

                        prob_metabo = de.ODEProblem(metabolism, vm, tspan_m, param)

                        prnt.append_to_file(
                            config.param_out_file,
                            [c_gid, idxm, *param, cells_volumes[c_gid]],
                            rank,
                        )

                        log_stage(f"checking param")
                        err_msg = utils.check_param(param, idxm)
                        if len(err_msg):
                            logging.info(err_msg)
                            log_stage(f"skip metabolism for c_gid: {c_gid}")
                            failed_cells.add(c_gid)
                            continue

                        log_stage("solve metabolism")

                        sol = None
                        error_solver = None

                        try:
                            sol = de.solve(
                                prob_metabo,
                                de.Rosenbrock23(autodiff=False),
                                reltol=1e-8,
                                abstol=1e-8,
                                saveat=1,
                                maxiters=1e6,
                            )

                            if sol.retcode != "Success":
                                print(f"sol.retcode: {sol.retcode}")

                        except Exception as e:
                            prnt.append_to_file(config.err_solver_output, c_gid, rank)
                            error_solver = e
                            failed_cells.add(c_gid)

                        if sol is None:
                            raise error_solver

                        um[(idxm + 1, c_gid)] = sol.u[-1]
                        prnt.append_to_file(
                            config.um_out_file, [c_gid, idxm, sol.u[-1]], rank
                        )

                        # u stands for Julia ODE var and m stands for metabolism
                        atpi_weighted_mean = um[(idxm + 1, c_gid)][
                            22
                        ]  # 0.5 * 1.2 + 0.5 * um[(idxm + 1, c_gid)][22] #um[(idxm+1,c_gid)][27]

                        adpi_weighted_mean = (
                                atpi_weighted_mean
                                / 2
                                * (
                                        -0.92
                                        + np.sqrt(
                                    0.92 * 0.92
                                    + 4 * 0.92 * (ATDPtot_n / atpi_weighted_mean - 1)
                                )
                                )
                        )  # 0.5 * 6.3e-3 + 0.5 * um[(idxm + 1, c_gid)][29] # um[(idxm+1,c_gid)][29]

                        #                         nao_weighted_mean = 0.5 * 140.0 + 0.5 * (
                        #                             140.0 - 1.33 * (um[(idxm + 1, c_gid)][6] - 10.0)
                        #                         )  # 140.0 - 1.33*(param[3] - 10.0) #14jan2021  # or 140.0 - .. # 144  # param[3] because pyhton indexing is 0,1,2.. julia is 1,2,..

                        ko_weighted_mean = (
                                0.5 * um[(0, c_gid)][95] + 0.5 * um[(idxm + 1, c_gid)][95]

                        )  # um[(idxm+1,c_gid)][7]

                        #                         nai_weighted_mean = (
                        #                             0.5 * 10.0 + 0.5 * um[(idxm + 1, c_gid)][6]
                        #                         )  # 0.5*10.0 + 0.5*um[(idxm+1,c_gid)][6] #um[(idxm+1,c_gid)][6]

                        #                         ki_weighted_mean = 0.5 * 140.0 + 0.5 * param[4]  # 14jan2021
                        #                         # feedback loop to constrain ndamus by metabolism output

                        um[(idxm, c_gid)] = None

                        log_stage("feedback")

                        # for seg in neurodamus_utils.gen_segs(nc, config.seg_filter):
                        #     seg.nao = nao_weighted_mean  # 140
                        #     seg.nai = nai_weighted_mean  # 10
                        #     seg.ko = ko_weighted_mean  # 5
                        #     seg.ki = ki_weighted_mean  # 140
                        #     seg.atpi = atpi_weighted_mean  # 1.4
                        #     seg.adpi = adpi_weighted_mean  # 0.03

                        # for seg in neurodamus_utils.gen_segs(nc, [Na.nai_var]):
                        #     seg.nao = nao_weighted_mean  # 140                       # TMP disable feedback from STEPS to NDAM

                        #                             for seg in neurodamus_utils.gen_segs(nc, [Na.nai_var]):
                        #                                 seg.nai = nai_weighted_mean  # 10

                        for seg in neurodamus_utils.gen_segs(nc, [config.KK.ki_var]):
                            seg.ko = ko_weighted_mean  # 5                           # TMP disable feedback from STEPS to NDAM

                        #                             for seg in neurodamus_utils.gen_segs(nc, [K.ki_var]):
                        #                                 seg.ki = ki_weighted_mean  # 140

                        for seg in neurodamus_utils.gen_segs(
                                nc, [config.ATP.atpi_var]
                        ):
                            seg.atpi = atpi_weighted_mean  # 1.4

                        for seg in neurodamus_utils.gen_segs(
                                nc, [config.ADP.adpi_var]
                        ):
                            seg.adpi = (
                                    atpi_weighted_mean
                                    / 2
                                    * (
                                            -0.92
                                            + np.sqrt(
                                        0.92 * 0.92
                                        + 4
                                        * 0.92
                                        * (ATDPtot_n / atpi_weighted_mean - 1)
                                    )
                                    )
                            )  # adpi_weighted_mean  # 0.03

                    comm.Barrier()
                    logging.info(
                        f"metabolism, remove failed cells: {', '.join([str(i) for i in failed_cells])}"
                    )
                    for i in failed_cells:
                        print("failed_cells:", i, "at idxm: ", idxm)
                        gid_to_cell.pop(i)

                    idxm += 1

            if (steps % config.dt_nrn2dt_bf == 0) and config.with_bloodflow:
                bf_manager.sync(ndamus)
                bf_manager.update_static_flow()
                if rank == 0:
                    bf_Fin = bf2n_flow.dot(bf_manager.graph.edge_properties["flow"].to_numpy())
                    bf_vol = bf2n_vol.dot(bf_manager.graph.edge_properties.volume.to_numpy())
                    print("TODO connect this to metab")
                    print("Fin per gid:", bf_Fin)
                    print("blood vol per gid:", bf_vol)

            rss.append(
                psutil.Process().memory_info().rss / (1024 ** 2)
            )  # memory consumption in MB

    ndamus.spike2file(os.getcwd() + '/' + prnt.file_path("out.dat"))
    TimerManager.timeit_show_stats()


if __name__ == "__main__":
    main()
