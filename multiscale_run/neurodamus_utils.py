from neurodamus.connection_manager import SynapseRuleManager
from mpi4py import MPI as MPI4PY

comm = MPI4PY.COMM_WORLD
from . import utils

import logging

import os

import config


def get_cell_volumes(ndamus):
    return {
        int(nc.CCell.gid): sum(
            [val.volume() for sublist in nc.CCell.all for val in sublist.allseg()]
        )
        for nc in ndamus.circuits.base_cell_manager.cells
    }

def get_cell_areas(ndamus):
    return {
        int(nc.CCell.gid): sum(
            [val.area() for sublist in nc.CCell.all for val in sublist.allseg()]
        )
        for nc in ndamus.circuits.base_cell_manager.cells
    }


def release_sums(release):
    sum_t = {}
    for s in release:
        for t in release[s]:
            sum_t.setdefault(t, 0.0)
            sum_t[t] += release[s][t]

    return sum_t


def collect_gaba_glutamate_releases(ndamus):
    collected_num_releases_glutamate, collected_num_releases_gaba = {}, {}

    synapse_managers = [
        manager
        for manager in ndamus.circuits.base_cell_manager.connection_managers.values()
        if isinstance(manager, SynapseRuleManager)
    ]

    for syn_manager in synapse_managers:
        for conn in syn_manager.all_connections():
            num_releases_glutamate = 0  # 12jan2021
            num_releases_gaba = 0  # 12jan2021
            if conn.sgid in config.target_gids:  # 12jan2021
                collected_num_releases_glutamate.setdefault(conn.sgid, {})
                collected_num_releases_gaba.setdefault(conn.sgid, {})

                for syn in conn._synapses:
                    if hasattr(syn, "A_AMPA_step"):
                        num_releases_glutamate += syn.release_accumulator
                        syn.release_accumulator = 0.0
                    elif hasattr(syn, "A_GABAA_step"):
                        num_releases_gaba += syn.release_accumulator
                        syn.release_accumulator = 0.0
                if isinstance(conn.sgid, float) or isinstance(conn.tgid, float):
                    raise Exception(
                        f"Rank {comm.Get_rank()} ids {conn.sgid} {conn.tgid} have floats!"
                    )

                collected_num_releases_glutamate[conn.sgid][
                    conn.tgid
                ] = num_releases_glutamate
                collected_num_releases_gaba[conn.sgid][conn.tgid] = num_releases_gaba

    return collected_num_releases_gaba, collected_num_releases_glutamate


def collect_received_events(releases, all_needs, gid_to_cell, outs_r):
    comm.Barrier()
    rank = comm.Get_rank()

    all_events = comm.reduce(releases, op=utils.add_dict, root=0)
    if rank == 0:
        for r, needs in all_needs.items():
            events = {s: all_events[s] for s in needs if s in all_events}
            comm.send(events, dest=r)
        received_events = {s: all_events[s] for s in gid_to_cell if s in all_events}
    else:
        received_events = comm.recv(source=0)
    comm.Barrier()

    all_outs_r = {}
    if rank == 0:
        for sgid, tv in all_events.items():
            for tgid, v in tv.items():
                all_outs_r[sgid] = all_outs_r.get(sgid, 0.0) + v

    for s, tv in received_events.items():
        releases.setdefault(s, {})
        for t, v in tv.items():
            if t in gid_to_cell:
                continue
            releases[s][t] = v
    comm.Barrier()
    # del received_events

    for sgid, tv in releases.items():
        for tgid, v in tv.items():
            outs_r[sgid] = outs_r.get(sgid, 0.0) + v

    return all_outs_r


def gen_ncs(gid_to_cell):
    for c_gid, nc in gid_to_cell.items():
        if c_gid not in config.target_gids:
            logging.warning(f"{c_gid} not in target_gids")

            for i in config.target_gids:
                print(i, type(i))

            continue

        yield c_gid, nc


def gen_segs(nc, filter0):
    for sec in nc.CCell.all:
        if not all(hasattr(sec, i) for i in filter0):
            continue

        for seg in sec.allseg():
            yield seg


def get_current_density(gid_to_cell, seg_filter):
    current_density = {}
    gids_without_valid_segs = set()
    for c_gid, nc in gen_ncs(gid_to_cell):
        _exhausted = object()
        if next(gen_segs(nc, [seg_filter]), _exhausted) is _exhausted:
            gids_without_valid_segs.add(c_gid)
        else:
            tot_area = sum(i.area() for i in gen_segs(nc, [seg_filter]))
            current_density[c_gid] = (
                sum(
                    getattr(seg, seg_filter) * seg.area()
                    for seg in gen_segs(nc, [seg_filter])
                )
                / tot_area
            )
    return current_density, gids_without_valid_segs


def get_current_mean(gid_to_cell, seg_filter):
    current_mean = {}
    gids_without_valid_segs = set()
    for c_gid, nc in gen_ncs(gid_to_cell):
        _exhausted = object()
        if next(gen_segs(nc, [seg_filter]), _exhausted) is _exhausted:
            gids_without_valid_segs.add(c_gid)
        else:
            tot_vol = sum(i.volume() for i in gen_segs(nc, [seg_filter]))
            current_mean[c_gid] = (
                sum(
                    getattr(seg, seg_filter) * seg.volume()
                    for seg in gen_segs(nc, [seg_filter])
                )
                / tot_vol
            )
    return current_mean, gids_without_valid_segs
