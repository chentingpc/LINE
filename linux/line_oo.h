// This file contains class declaration and utility functions

// Format of the training file:
//
// The training file contains serveral lines, each line represents a DIRECTED edge in the network.
// More specifically, each line has the following format "<u> <v> <w>", meaning an edge from <u> to <v> with weight as <w>.
// <u> <v> and <w> are seperated by ' ' or '\t' (blank or tab)
// For UNDIRECTED edge, the user should use two DIRECTED edges to represent it.

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>

using namespace std;

// ?? debug
#define NEG_SAMPLING_POWER 0.75
const int hash_table_size = 30000000;
const int neg_table_size = 1e8;


typedef float real;                   // Precision of float numbers
const real LOG_MIN = 1e-8;            // Smoother for log
#define MAX_STRING 2000

struct Vertex {
  double degree;
  char *name;
};

class DataHelper {
  struct Vertex           *vertex;
  int                     *vertex_hash_table;

  double                  *edge_weight;
  int                     *edge_source_id;
  int                     *edge_target_id;

  int                     num_vertices;
  long long               num_edges;
  int                     max_num_vertices;

  /* Build a hash table, mapping each vertex name to a unique vertex id */
  unsigned int hash(char *key) {
    unsigned int seed = 131;
    unsigned int hash = 0;
    while (*key) {
      hash = hash * seed + (*key++);
    }
    return hash % hash_table_size;
  }

  void init_hash_table() {
    vertex_hash_table = (int *)malloc(hash_table_size * sizeof(int));
    for (int k = 0; k != hash_table_size; k++)
      vertex_hash_table[k] = -1;
  }

  void insert_hash_table(char *key, int value) {
    int addr = hash(key);
    while (vertex_hash_table[addr] != -1)
      addr = (addr + 1) % hash_table_size;
    vertex_hash_table[addr] = value;
  }

  int search_hash_table(char *key) {
    int addr = hash(key);
    while (1) {
      if (vertex_hash_table[addr] == -1)
        return -1;
      if (!strcmp(key, vertex[vertex_hash_table[addr]].name))
        return vertex_hash_table[addr];
      addr = (addr + 1) % hash_table_size;
    }
    return -1;
  }

  int add_vertex(char *name) {
    int length = strlen(name) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    vertex[num_vertices].name = (char *)calloc(length, sizeof(char));
    strcpy(vertex[num_vertices].name, name);
    vertex[num_vertices].degree = 0;
    num_vertices++;
    if (num_vertices + 2 >= max_num_vertices) {
      max_num_vertices += 10000;
      vertex = (struct Vertex *)realloc(vertex, max_num_vertices * sizeof(struct Vertex));
    }
    insert_hash_table(name, num_vertices - 1);
    return num_vertices - 1;
  }

  /*
   * Read network from the training file
   * Outcomes: vertex, edge_weight, ...
   */
  void read_data(string network_file) {
    FILE *fin;
    char name_v1[MAX_STRING], name_v2[MAX_STRING], str[2 * MAX_STRING + 10000];
    int vid;
    double weight;

    fin = fopen(network_file.c_str(), "rb");
    if (fin == NULL) {
      printf("ERROR: network file not found!\n");
      exit(1);
    }
    num_edges = 0;
    while (fgets(str, sizeof(str), fin)) num_edges++;
    fclose(fin);
    printf("Number of edges: %lld          \n", num_edges);

    edge_source_id = (int *)malloc(num_edges*sizeof(int));
    edge_target_id = (int *)malloc(num_edges*sizeof(int));
    edge_weight = (double *)malloc(num_edges*sizeof(double));
    if (edge_source_id == NULL || edge_target_id == NULL || edge_weight == NULL) {
      printf("Error: memory allocation failed!\n");
      exit(1);
    }

    vertex = (struct Vertex *)calloc(max_num_vertices, sizeof(struct Vertex));

    fin = fopen(network_file.c_str(), "rb");
    num_vertices = 0;
    for (long long k = 0; k != num_edges; k++) {
      fscanf(fin, "%s %s %lf", name_v1, name_v2, &weight);

      if (k % 10000 == 0) {
        printf("Reading edges: %.3lf%%%c", k / (double)(num_edges + 1) * 100, 13);
        fflush(stdout);
      }

      vid = search_hash_table(name_v1);
      if (vid == -1) vid = add_vertex(name_v1);
      vertex[vid].degree += weight;
      edge_source_id[k] = vid;

      vid = search_hash_table(name_v2);
      if (vid == -1) vid = add_vertex(name_v2);
      vertex[vid].degree += weight;
      edge_target_id[k] = vid;

      edge_weight[k] = weight;
    }
    fclose(fin);
    printf("Number of vertices: %d          \n", num_vertices);
  }

 public:
  explicit DataHelper(string network_file) :
      num_vertices(0),
      max_num_vertices(10000),
      num_edges(0) {
    init_hash_table();
    read_data(network_file);
  }

  int get_num_vertices() {
    return num_vertices;
  }

  int get_num_edges() {
    return num_edges;
  }

  const struct Vertex * get_vertex() {
    return vertex;
  }

  const double * get_edge_weight() {
    return edge_weight;
  }

  const int * get_edge_source_id() {
    return edge_source_id;
  }

  const int * get_edge_target_id() {
    return edge_target_id;
  }
};


/* high precision unifrom distribution generator */
class GSLRandUniform {
  const gsl_rng_type    *gsl_T;
  gsl_rng               *gsl_r;

 public:
  explicit GSLRandUniform(const int seed = 314159265) {
    gsl_rng_env_setup();
    gsl_T = gsl_rng_rand48;
    gsl_r = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r, seed);
  }

  double operator()() {
    // This function returns a double precision floating point number
    //  uniformly distributed in the range [0,1)
    return gsl_rng_uniform(gsl_r);
  }
};


class EdgeSampler {
  const double            *edge_weight;
  const long long         num_edges;

  GSLRandUniform          gsl_rand;

  long long               *alias;
  double                  *prob;

  /* The alias sampling algorithm, which is used to sample an edge in O(1) time. */
  void init_alias_table() {
    alias = (long long *)malloc(num_edges*sizeof(long long));
    prob = (double *)malloc(num_edges*sizeof(double));
    if (alias == NULL || prob == NULL) {
      printf("Error: memory allocation failed!\n");
      exit(1);
    }

    double *norm_prob = (double*)malloc(num_edges*sizeof(double));
    long long *large_block = (long long*)malloc(num_edges*sizeof(long long));
    long long *small_block = (long long*)malloc(num_edges*sizeof(long long));
    if (norm_prob == NULL || large_block == NULL || small_block == NULL) {
      printf("Error: memory allocation failed!\n");
      exit(1);
    }

    double sum = 0;
    long long cur_small_block, cur_large_block;
    long long num_small_block = 0, num_large_block = 0;

    for (long long k = 0; k != num_edges; k++) sum += edge_weight[k];
    for (long long k = 0; k != num_edges; k++) norm_prob[k] = edge_weight[k] * num_edges / sum;

    for (long long k = num_edges - 1; k >= 0; k--) {
      if (norm_prob[k] < 1)
        small_block[num_small_block++] = k;
      else
        large_block[num_large_block++] = k;
    }

    while (num_small_block && num_large_block) {
      cur_small_block = small_block[--num_small_block];
      cur_large_block = large_block[--num_large_block];
      prob[cur_small_block] = norm_prob[cur_small_block];
      alias[cur_small_block] = cur_large_block;
      norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
      if (norm_prob[cur_large_block] < 1)
        small_block[num_small_block++] = cur_large_block;
      else
        large_block[num_large_block++] = cur_large_block;
    }

    while (num_large_block) prob[large_block[--num_large_block]] = 1;
    while (num_small_block) prob[small_block[--num_small_block]] = 1;

    free(norm_prob);
    free(small_block);
    free(large_block);
  }

 public:
  explicit EdgeSampler(const double *edge_weight, const long long num_edges) :
      edge_weight(edge_weight),
      num_edges(num_edges) {
    gsl_rand = GSLRandUniform();
    init_alias_table();
  }

  long long sample() {
    long long k = (long long)num_edges * gsl_rand();
    return gsl_rand() < prob[k] ? k : alias[k];
  }
};


class NodeSampler {
  const struct Vertex *vertex;
  int                 *neg_table;
  int                 num_vertices;

  /* Fastly generate a random integer */
  int Rand(unsigned long long &seed) {
    seed = seed * 25214903917 + 11;
    return (seed >> 16) % neg_table_size;
  }

  void init_neg_table() {
    double sum = 0, cur_sum = 0, por = 0;
    int vid = 0;
    neg_table = (int *)malloc(neg_table_size * sizeof(int));
    for (int k = 0; k != num_vertices; k++) sum += pow(vertex[k].degree, NEG_SAMPLING_POWER);
    for (int k = 0; k != neg_table_size; k++) {
      if ((double)(k + 1) / neg_table_size > por) {
        cur_sum += pow(vertex[vid].degree, NEG_SAMPLING_POWER);
        por = cur_sum / sum;
        vid++;
      }
      neg_table[k] = vid - 1;
    }
  }

 public:
  explicit NodeSampler (const struct Vertex *vertex, const int num_vertices) :
      vertex(vertex),
      num_vertices(num_vertices) {
    init_neg_table();
  }

  /* Sample negative vertex samples according to vertex degrees */
  long long sample(unsigned long long seed = 0) {
    return neg_table[Rand(seed)];
  };
};


/* Fastly compute sigmoid function */
#define SIGMOID_BOUND 6
const int sigmoid_table_size = 1000;

class Sigmoid {
  real *sigmoid_table;

  void init_sigmoid_table() {
    real x;
    sigmoid_table = (real *)malloc((sigmoid_table_size + 1) * sizeof(real));
    for (int k = 0; k != sigmoid_table_size; k++) {
      x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
      sigmoid_table[k] = 1 / (1 + exp(-x));
    }
  }

 public:
  Sigmoid() {
    init_sigmoid_table();
  }

  real operator()(real x) {
    if (x > SIGMOID_BOUND) return 1;
    else if (x < -SIGMOID_BOUND) return 0;
    int k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2;
    return sigmoid_table[k];
  }
};

class EmbeddingModel {
  int                     dim;
  int                     num_vertices;
  int                     order;
  int                     num_negative;
  real                    init_rho;
  real                    rho;
  long long               total_samples;
  int                     current_sample_count;
  int                     num_threads;

  const struct Vertex     *vertex;
  const int               *edge_source_id;
  const int               *edge_target_id;

  real                    *emb_vertex;
  real                    *emb_context;

  class Sigmoid           *sigmoid;
  class DataHelper        *data_helper;
  class NodeSampler       *node_sampler;
  class EdgeSampler       *edge_sampler;

  /* Initialize the vertex embedding and the context embedding */
  void init_vector() {
    long long a, b;

    a = posix_memalign((void **)&emb_vertex, 128, (long long)num_vertices * dim * sizeof(real));
    if (emb_vertex == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
    for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
      emb_vertex[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;

    a = posix_memalign((void **)&emb_context, 128, (long long)num_vertices * dim * sizeof(real));
    if (emb_context == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
    for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
      emb_context[a * dim + b] = 0;
  }

  /* Update embeddings & return likelihood */
  real update(real *vec_u, real *vec_v, real *vec_error, int label) {
    real x = 0, g;
    for (int c = 0; c != dim; c++) x += vec_u[c] * vec_v[c];
    g = (label - (*sigmoid)(x)) * rho;
    for (int c = 0; c != dim; c++) vec_error[c] += g * vec_v[c];
    for (int c = 0; c != dim; c++) vec_v[c] += g * vec_u[c];

    return label > 0? log(1-g+LOG_MIN): log(1+g+LOG_MIN);
  }

  static void *static_thread_entry(void* id) {
      EmbeddingModel* p = static_cast<EmbeddingModel*>(id);
      return p->train_thread(id);
  }

  void *train_thread(void *id) {
    long long u, v, lu, lv, target, label;
    long long count = 0, last_count = 0, ll_count = 0, curedge;
    unsigned long long seed = 0;//(long long)id; // debug
    real *vec_error = (real *)calloc(dim, sizeof(real));
    double ll = 0.;
    while (1) {
      if (count > total_samples / num_threads + 2) break;

      if (count - last_count>10000) {
        current_sample_count += count - last_count;
        last_count = count;
        printf("%cRho: %f  Progress: %.3lf%%, LogLikelihood %.9lf", 13, rho,
          (real)current_sample_count / (real)(total_samples + 1) * 100, ll / ll_count);
        fflush(stdout);
        rho = init_rho * (1 - current_sample_count / (real)(total_samples + 1));
        if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;

        ll = 0.;
        ll_count = 0;
      }

      curedge = edge_sampler->sample();
      u = edge_source_id[curedge];
      v = edge_target_id[curedge];

      lu = u * dim;
      for (int c = 0; c != dim; c++)
        vec_error[c] = 0;

      // NEGATIVE SAMPLING
      for (int d = 0; d != num_negative + 1; d++) {
        if (d == 0) {
          target = v;
          label = 1;
        } else {
          target = node_sampler->sample(seed);
          label = 0;
        }
        lv = target * dim;
        if (order == 1) ll += update(&emb_vertex[lu], &emb_vertex[lv], vec_error, label);
        if (order == 2) ll += update(&emb_vertex[lu], &emb_context[lv], vec_error, label);
      }
      for (int c = 0; c != dim; c++) emb_vertex[c + lu] += vec_error[c];

      count++;
      ll_count++;
    }
    free(vec_error);
    pthread_exit(NULL);
  }

 public:
  EmbeddingModel(DataHelper *data_helper,
                 NodeSampler *node_sampler,
                 EdgeSampler *edge_sampler,
                 int order, int dim) :
                data_helper(data_helper),
                node_sampler(node_sampler),
                edge_sampler(edge_sampler),
                order(order),
                dim(dim) {
    sigmoid = new Sigmoid();
    edge_source_id = data_helper->get_edge_source_id();
    edge_target_id = data_helper->get_edge_target_id();
    init_vector();
  }

  void train(int num_threads, int num_negative, long long total_samples, real init_rho) {
    long a;
    current_sample_count = 0;
    rho = init_rho;
    this->num_threads = num_threads;
    this->num_negative = num_negative;
    this->total_samples = total_samples;
    this->init_rho = init_rho;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));

    clock_t start = clock();
    printf("--------------------------------\n");
    long long f = 0;
    //train_thread(&f); // debug
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, static_thread_entry, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    printf("\n");
    clock_t finish = clock();
    printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);
  }

  void save(string embedding_file, bool is_binary) {
    FILE *fo = fopen(embedding_file.c_str(), "wb");
    fprintf(fo, "%d %d\n", num_vertices, dim);
    for (int a = 0; a < num_vertices; a++) {
      fprintf(fo, "%s ", vertex[a].name);
      if (is_binary) for (int b = 0; b < dim; b++) fwrite(&emb_vertex[a * dim + b], sizeof(real), 1, fo);
      else for (int b = 0; b < dim; b++) fprintf(fo, "%lf ", emb_vertex[a * dim + b]);
      fprintf(fo, "\n");
    }
    fclose(fo);
  }
};