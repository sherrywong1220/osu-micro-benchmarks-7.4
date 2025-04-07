#define BENCHMARK "OSU MPI_Get Multi%s Latency Test"
/*
 * Copyright (c) 2002-2024 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */

#include <osu_util_mpi.h>

double t_start = 0.0, t_end = 0.0;
char *rbuf = NULL, *win_base = NULL;
omb_graph_options_t omb_graph_op;
MPI_Comm omb_comm = MPI_COMM_NULL;

void print_multi_latency(int, int, double, int, struct omb_stat_t);
void run_pscw_with_multi(int, int, enum WINDOW);

int main(int argc, char *argv[])
{
    int rank, nprocs;
    int po_ret = PO_OKAY;
    omb_mpi_init_data omb_init_h;
#if MPI_VERSION >= 3
    options.win = WIN_ALLOCATE;
#else
    options.win = WIN_CREATE;
#endif

    options.bench = ONE_SIDED;
    options.subtype = LAT;
    options.synctype = ALL_SYNC;
    MPI_Datatype mpi_type_list[OMB_NUM_DATATYPES];
    set_header(HEADER);
    set_benchmark_name("osu_get_multi_lat");

    po_ret = process_options(argc, argv);
    omb_populate_mpi_type_list(mpi_type_list);
    if (options.validate) {
        OMB_ERROR_EXIT("Benchmark does not support validation");
    }

    if (PO_OKAY == po_ret && NONE != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    omb_init_h = omb_mpi_init(&argc, &argv);
    omb_comm = omb_init_h.omb_comm;
    if (MPI_COMM_NULL == omb_comm) {
        OMB_ERROR_EXIT("Cant create communicator");
    }
    MPI_CHECK(MPI_Comm_rank(omb_comm, &rank));
    MPI_CHECK(MPI_Comm_size(omb_comm, &nprocs));
    if (0 == rank) {
        if (options.omb_dtype_itr > 1 || mpi_type_list[0] != MPI_CHAR) {
            fprintf(stderr, "Benchmark supports only MPI_CHAR. Continuing with "
                            "MPI_CHAR.\n");
            fflush(stderr);
        }
    }

    if (0 == rank) {
        switch (po_ret) {
            case PO_CUDA_NOT_AVAIL:
                fprintf(stderr, "CUDA support not enabled.  Please recompile "
                                "benchmark with CUDA support.\n");
                break;
            case PO_OPENACC_NOT_AVAIL:
                fprintf(stderr, "OPENACC support not enabled.  Please "
                                "recompile benchmark with OPENACC support.\n");
                break;
            case PO_BAD_USAGE:
                print_bad_usage_message(rank);
            case PO_HELP_MESSAGE:
                usage_one_sided("osu_get_multi_lat");
                break;
            case PO_VERSION_MESSAGE:
                print_version_message(rank);
                omb_mpi_finalize(omb_init_h);
                exit(EXIT_SUCCESS);
            case PO_OKAY:
                break;
        }
    }

    switch (po_ret) {
        case PO_CUDA_NOT_AVAIL:
        case PO_OPENACC_NOT_AVAIL:
        case PO_BAD_USAGE:
            omb_mpi_finalize(omb_init_h);
            exit(EXIT_FAILURE);
        case PO_HELP_MESSAGE:
        case PO_VERSION_MESSAGE:
            omb_mpi_finalize(omb_init_h);
            exit(EXIT_SUCCESS);
        case PO_OKAY:
            break;
    }

    // 确保进程数是偶数
    if (nprocs % 2 != 0) {
        if (rank == 0) {
            fprintf(stderr, "This test requires an even number of processes\n");
        }
        omb_mpi_finalize(omb_init_h);
        return EXIT_FAILURE;
    }

    options.pairs = nprocs / 2;
    
    if (rank == 0) {
        fprintf(stdout, "# OSU MPI_Get Multi Latency Test v%s\n", PACKAGE_VERSION);
        fprintf(stdout, "# Pairs: %d\n", options.pairs);
        fprintf(stdout, "# Window: %s\n", win_info[options.win]);
        fprintf(stdout, "# Synchronization: PSCW\n");
        fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Latency (us)");
        fflush(stdout);
    }

    run_pscw_with_multi(rank, options.pairs, options.win);

    omb_mpi_finalize(omb_init_h);

    if (NONE != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    return EXIT_SUCCESS;
}

void print_multi_latency(int rank, int size, double t_total, int iterations, struct omb_stat_t omb_stat)
{
    if (rank == 0) {
        double latency = (t_total * 1e6) / iterations;
        fprintf(stdout, "%-*d%*.*f", 10, size, FIELD_WIDTH, FLOAT_PRECISION, latency);
        if (options.omb_tail_lat) {
            OMB_ITR_PRINT_STAT(omb_stat.res_arr);
        }
        fprintf(stdout, "\n");
        fflush(stdout);
    }
}

void run_pscw_with_multi(int rank, int pairs, enum WINDOW type)
{
    int size, i, target;
    double t_total = 0.0, t_graph_start = 0.0, t_graph_end = 0.0;
    int papi_eventset = OMB_PAPI_NULL;
    omb_graph_data_t *omb_graph_data = NULL;
    MPI_Aint disp = 0;
    MPI_Win my_win;
    char *my_rbuf = NULL, *my_win_base = NULL;
    MPI_Group comm_group, group;
    MPI_CHECK(MPI_Comm_group(omb_comm, &comm_group));
    int *ranks = NULL;
    double *omb_lat_arr = NULL;
    struct omb_stat_t omb_stat;
    MPI_Comm barrier_comm;
    double t_total_reduce = 0.0;

    // 为每组进程创建独立的通信子，用于同步
    MPI_CHECK(MPI_Comm_split(omb_comm, rank < pairs, 0, &barrier_comm));

    if (options.omb_tail_lat) {
        omb_lat_arr = malloc(options.iterations * sizeof(double));
        OMB_CHECK_NULL_AND_EXIT(omb_lat_arr, "Unable to allocate memory");
    }
    
    omb_papi_init(&papi_eventset);
    
    for (size = options.min_message_size; size <= options.max_message_size;
         size = (size ? size * 2 : 1)) {
        
        // 为每个进程分配内存
        allocate_memory_one_sided(rank, &my_rbuf, &my_win_base, size, type, &my_win);

#if MPI_VERSION >= 3
        if (type == WIN_DYNAMIC) {
            disp = disp_remote;
        }
#endif

        if (size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }

        omb_graph_allocate_and_get_data_buffer(&omb_graph_data, &omb_graph_op,
                                               size, options.iterations);
        t_total = 0.0;
        
        if (rank < pairs) {
            // 发起方进程
            target = rank + pairs;
            
            // 创建包含目标进程的组
            ranks = malloc(sizeof(int));
            ranks[0] = target;
            MPI_CHECK(MPI_Group_incl(comm_group, 1, ranks, &group));
            
            for (i = 0; i < options.skip + options.iterations; i++) {
                MPI_CHECK(MPI_Barrier(omb_comm));
                
                if (i == options.skip) {
                    omb_papi_start(&papi_eventset);
                    t_start = MPI_Wtime();
                }
                
                if (i >= options.skip) {
                    t_graph_start = MPI_Wtime();
                }
                
                // 启动访问窗口
                MPI_CHECK(MPI_Win_start(group, 0, my_win));
                
                // 执行Get操作
                MPI_CHECK(MPI_Get(my_rbuf, size, MPI_CHAR, target, disp, size, MPI_CHAR, my_win));
                
                // 完成访问窗口
                MPI_CHECK(MPI_Win_complete(my_win));
                
                // 暴露本地窗口
                MPI_CHECK(MPI_Win_post(group, 0, my_win));
                
                // 等待远程访问完成
                MPI_CHECK(MPI_Win_wait(my_win));
                
                if (i >= options.skip) {
                    t_graph_end = MPI_Wtime();
                    if (options.omb_tail_lat) {
                        omb_lat_arr[i - options.skip] = (t_graph_end - t_graph_start) * 1.0e6 / 2.0;
                    }
                    if (options.graph) {
                        omb_graph_data->data[i - options.skip] = (t_graph_end - t_graph_start) * 1.0e6 / 2.0;
                    }
                }
            }
            
            t_end = MPI_Wtime();
            t_total = t_end - t_start;
            free(ranks);
            
        } else if (rank < 2 * pairs) {
            // 目标进程
            target = rank - pairs;
            
            // 创建包含源进程的组
            ranks = malloc(sizeof(int));
            ranks[0] = target;
            MPI_CHECK(MPI_Group_incl(comm_group, 1, ranks, &group));
            
            for (i = 0; i < options.skip + options.iterations; i++) {
                MPI_CHECK(MPI_Barrier(omb_comm));
                
                if (i == options.skip) {
                    omb_papi_start(&papi_eventset);
                }
                
                // 暴露本地窗口
                MPI_CHECK(MPI_Win_post(group, 0, my_win));
                
                // 等待远程访问完成
                MPI_CHECK(MPI_Win_wait(my_win));
                
                // 启动访问窗口
                MPI_CHECK(MPI_Win_start(group, 0, my_win));
                
                // 执行Get操作
                MPI_CHECK(MPI_Get(my_rbuf, size, MPI_CHAR, target, disp, size, MPI_CHAR, my_win));
                
                // 完成访问窗口
                MPI_CHECK(MPI_Win_complete(my_win));
            }
            
            free(ranks);
        }
        
        MPI_CHECK(MPI_Barrier(omb_comm));
        
        // 汇总所有发起方的时间
        MPI_CHECK(MPI_Reduce(&t_total, &t_total_reduce, 1, MPI_DOUBLE, MPI_SUM, 0, barrier_comm));
        t_total = t_total_reduce / pairs;
        
        omb_stat = omb_calculate_tail_lat(omb_lat_arr, rank, 1);
        omb_papi_stop_and_print(&papi_eventset, size);
        print_multi_latency(rank, size, t_total, options.iterations, omb_stat);
        
        if (options.graph && 0 == rank) {
            omb_graph_data->avg = (t_total * 1.0e6) / options.iterations;
        }
        
        if (options.graph) {
            omb_graph_plot(&omb_graph_op, benchmark_name);
        }
        
        MPI_CHECK(MPI_Group_free(&group));
        free_memory_one_sided(my_rbuf, my_win_base, type, my_win, rank);
    }
    
    MPI_CHECK(MPI_Comm_free(&barrier_comm));
    omb_graph_combined_plot(&omb_graph_op, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_op);
    omb_papi_free(&papi_eventset);
    free(omb_lat_arr);
    MPI_CHECK(MPI_Group_free(&comm_group));
}
/* vi: set sw=4 sts=4 tw=80: */ 