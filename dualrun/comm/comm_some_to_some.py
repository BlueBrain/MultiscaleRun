#
# test need to be run with 4 ranks
# mpirun -n 4 python3 comm_some_to_some.py
#

from mpi4py import MPI
import numpy as np

def init_send_mask(itype):
    if myrank == 0:
        send_mask = np.array([0,0,1,1], dtype=itype)
        # print("I'm {0} send list is: {1}".format(myrank, send_mask))
    elif myrank == 1:
        send_mask = np.array([1,0,1,0], dtype=itype)
        # print("I'm {0} send list is: {1}".format(myrank, send_mask))
    elif myrank == 2:
        send_mask = np.array([1,0,0,1], dtype=itype)
        # print("I'm {0} send list is: {1}".format(myrank, send_mask))
    else:
        send_mask = np.zeros(nproc, dtype=itype)
        # print("I'm {0} send list is: {1}".format(myrank, send_mask))
    return send_mask

def check_recv_mask(recv_mask):
    if myrank == 0:
        assert recv_mask[0] == 0
        assert recv_mask[1] == 1
        assert recv_mask[2] == 1
        assert recv_mask[3] == 0
    elif myrank == 1:
        assert recv_mask[0] == 0
        assert recv_mask[1] == 0
        assert recv_mask[2] == 0
        assert recv_mask[3] == 0
    elif myrank == 2:
        assert recv_mask[0] == 1
        assert recv_mask[1] == 1
        assert recv_mask[2] == 0
        assert recv_mask[3] == 0
    else:
        assert recv_mask[0] == 1
        assert recv_mask[1] == 0
        assert recv_mask[2] == 1
        assert recv_mask[3] == 0

def get_mpi_int_type(itype):
    if itype == np.int64:
        return MPI.LONG
    elif itype == np.int32:
        return MPI.INT
    elif itype == np.int16:
        return MPI.SHORT
    elif itype == np.int8:
        return MPI.CHAR
    else:
        raise "not know np int type"

def scatter_mask(send_mask, recv_mask, comm):
    # check preconditions
    itype = send_mask.dtype
    assert(itype == recv_mask.dtype)
    nproc = comm.Get_size()
    assert(nproc < np.iinfo(itype).max)

    mpi_int = get_mpi_int_type(itype)
    for i in range(0, nproc):
        buff = np.zeros(1, dtype=itype)
        comm.Scatter([send_mask, 1, mpi_int],[buff, 1, mpi_int], root=i)
        recv_mask[i] = buff[0]

def alltoallv_mask(send_mask, recv_mask, comm):
    # check preconditions
    itype = send_mask.dtype
    assert(itype == recv_mask.dtype)
    nproc = comm.Get_size()
    assert(nproc < np.iinfo(itype).max)
    s_counts = np.zeros(nproc, dtype=itype)
    r_counts = np.zeros(nproc, dtype=itype)
    s_displs = np.zeros(nproc, dtype=itype)
    r_displs = np.zeros(nproc, dtype=itype)
    disp = 0
    size = np.iinfo(itype).bits/8
    for i in range (nproc):
        s_counts[i] = r_counts[i] = size
        s_displs[i] = r_displs[i] = disp
        disp += size
    s_msg = [send_mask, (s_counts, s_displs), MPI.BYTE]
    r_msg = [recv_mask, (r_counts, r_displs), MPI.BYTE]
    comm.Alltoallv(s_msg, r_msg)

# return a list of ranks to which we need to send tets info and a list from which
# we need to receive tets info
def get_sendRecvTet_map(neurSecmap, tet2host, comm):

    rank = comm.Get_rank()
    nRanks = comm.Get_size()

    # list containing, for each rank, the list of tets we need to communicate
    sendTetArray = [np.array([], dtype=np.int64) for i in range(nRanks)]
    recvTetArray = [np.array([], dtype=np.int64) for i in range(nRanks)]

    # get send array
    for secmap in neurSecmap:
        for sec, tet2fraction_map in secmap:
            for segment in tet2fraction_map:
                for tet, frac in segment:
                    rank2comm = tet2host[tet]
                    # skip self comm
                    if rank2comm == rank:
                        continue
                    # store list of tets
                    is_there_tet = False
                    for tetstore in sendTetArray[rank2comm]:
                        if (tetstore == tet):
                            is_there_tet = True
                    if not is_there_tet:
                        sendTetArray[rank2comm] = np.append(sendTetArray[rank2comm], tet)

    # # print out for debug purposes
    # for r in range(0, size):
    #     if sendTetArray[r].shape[0] > 0:
    #         print("rank ", rank, "send to rank", r, "tets ",  sendTetArray[r])

    # send/recv tet #
    sendMask = np.zeros(nRanks, dtype=np.int16)
    recvMask = np.zeros(nRanks, dtype=np.int16)

    for r in range(0, nRanks):
        sendMask[r] = sendTetArray[r].shape[0]

    # debug
    # print("rank ", rank, "sendMask ", sendMask)

    # scatter_mask(sendMask, recvMask, comm)
    alltoallv_mask(sendMask, recvMask, comm)

    #debug
    # print("rank ", rank, "recvMask ", recvMask)

    # build recv list
    req = [MPI.REQUEST_NULL  for r in range(0,nRanks)]
    for r in range(0, nRanks):
        if r == rank:
            continue
        # post isend
        nSend = sendMask[r]
        if nSend > 0:
            comm.Isend([sendTetArray[r], nSend, MPI.LONG], dest=r, tag=r)
        # post irecv
        nRecv = recvMask[r]
        if nRecv > 0:
            recvTetArray[r] = np.zeros(nRecv, dtype=np.int64)
            req[r] = comm.Irecv([recvTetArray[r], nRecv, MPI.LONG], source=r, tag=MPI.ANY_TAG)
    # wait for all
    MPI.Request.Waitall(req)

    # # print out for debug purposes
    # for r in range(0, nRanks):
    #     if recvTetArray[r].shape[0] > 0:
    #         print("rank ", rank, "recv from rank", r, "tets ",  recvTetArray[r])

    return sendTetArray, recvTetArray


def reduce_tagged_only(tet_currents, tet_currents_all, sendTetArray, \
    recvTetArray, comm):

    rank = comm.Get_rank()
    nRanks = comm.Get_size()
    # recv send buff
    buff_s = [np.array([], dtype=float) for i in range(nRanks)]
    buff_r = [np.array([], dtype=float) for i in range(nRanks)]

    req = [MPI.REQUEST_NULL  for r in range(0, nRanks)]

    to_be_done = []

    for r in range(0, nRanks):
        if r == rank:
            continue
        # post isend
        nTetSend = int(sendTetArray[r].shape[0])
        if nTetSend > 0:
            # build buffer
            buff_s[r] = np.zeros(nTetSend, dtype=float)
            for tet in range(0, nTetSend):
                buff_s[r][tet] = tet_currents[sendTetArray[r][tet]]
            comm.Isend([buff_s[r], nTetSend, MPI.DOUBLE], dest=r, tag=r)

        # post irecv
        nTetRecv = int(recvTetArray[r].shape[0])
        if nTetRecv > 0:
            buff_r[r] = np.zeros(nTetRecv, dtype=float)
            req[r] = comm.Irecv([buff_r[r], nTetRecv, MPI.DOUBLE], source=r, tag=MPI.ANY_TAG)
            to_be_done.append(r)

    # init store
    tet_currents_all = np.array(tet_currents)
    # add buff when ready
    while (len(to_be_done) > 0):
        for r in list(to_be_done):
            if (req[r].test()[0]):
                nTetRecv = recvTetArray[r].shape[0]
                for tet in range(0, nTetRecv):
                    tet_currents_all[recvTetArray[r][tet]] += buff_r[r][tet]
                to_be_done.remove(r)
            else:
                continue


def check_comm_against_allreduce(tet_currents, tet_currents_all, tet2host, comm):
    rank = comm.Get_rank()
    ntets = tet_currents.shape[0]
    tet_currents_all_ref = np.zeros((ntets,), dtype=float)
    comm.Allreduce(tet_currents, tet_currents_all_ref, op=MPI.SUM)
    for tet in range(0, ntets):
        if tet2host[tet] == rank:
            if tet_currents_all[tet] != tet_currents_all_ref[tet]:
                a = tet_currents_all[tet]
                b = tet_currents_all_ref[tet]
                tol = 4 * max(abs(a), abs(b)) * np.finfo(float).eps
                err = abs(a-b)
                assert err > tol, "new comm err"


# simple unit test for mask scattering
if(__name__ =='__main__'):
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    nproc = comm.Get_size()
    assert nproc == 4

    dt = np.int8

    # scatter method
    if (myrank ==0):
        print("--- test scatter method ---")
    send_mask = init_send_mask(dt)
    recv_mask = np.zeros(nproc, dtype=dt)
    scatter_mask(send_mask, recv_mask, comm)
    check_recv_mask(recv_mask)

    # print("After Scatter, I'm {0} and recv list is: {1}".format(myrank, recv_mask))

    if (myrank ==0):
        print("--- test Alltoallv method ---")
    recv_mask = np.zeros(nproc, dtype=dt)
    alltoallv_mask(send_mask, recv_mask, comm)
    check_recv_mask(recv_mask)