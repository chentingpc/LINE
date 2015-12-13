
#include "./line_oo.h"

string network_file, embedding_file;
int is_binary = 0, num_threads = 1, order = 2, dim = 100, num_negative = 5;
long long total_samples = 1;
real init_rho = 0.025;

void train() {
  printf("--------------------------------\n");
  printf("Order: %d\n", order);
  printf("Threads: %d\n", num_threads);
  printf("Samples: %lldM\n", total_samples / 1000000);
  printf("Negative: %d\n", num_negative);
  printf("Dimension: %d\n", dim);
  printf("Initial rho: %lf\n", init_rho);
  printf("--------------------------------\n");

  DataHelper data_helper = DataHelper(network_file);
  NodeSampler node_sampler = NodeSampler(data_helper.get_vertex(),
                                         data_helper.get_num_vertices());
  EdgeSampler edge_sampler = EdgeSampler(data_helper.get_edge_weight(),
                                         data_helper.get_num_edges());
  EmbeddingModel model = EmbeddingModel(&data_helper, &node_sampler, &edge_sampler, order, dim);
  model.train(num_threads, num_negative, total_samples, init_rho);
  model.save(embedding_file, is_binary);
}

int arg_pos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("LINE: Large Information Network Embedding\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse network data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the learnt embeddings\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the learnt embeddings in binary moded; default is 0 (off)\n");
    printf("\t-size <int>\n");
    printf("\t\tSet dimension of vertex embeddings; default is 100\n");
    printf("\t-order <int>\n");
    printf("\t\tThe type of the model; 1 for first order, 2 for second order; default is 2\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5\n");
    printf("\t-samples <int>\n");
    printf("\t\tSet the number of training samples as <int>Million; default is 1\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 1)\n");
    printf("\t-rho <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\nExamples:\n");
    printf("./line -train net.txt -output vec.txt -binary 1 -size 200 -order 2 -negative 5 -samples 100 -rho 0.025 -threads 20\n\n");
    return 0;
  }
  if ((i = arg_pos((char *)"-train", argc, argv)) > 0) network_file = string(argv[i + 1]);
  if ((i = arg_pos((char *)"-output", argc, argv)) > 0) embedding_file = string(argv[i + 1]);
  if ((i = arg_pos((char *)"-binary", argc, argv)) > 0) is_binary = atoi(argv[i + 1]);
  if ((i = arg_pos((char *)"-size", argc, argv)) > 0) dim = atoi(argv[i + 1]);
  if ((i = arg_pos((char *)"-order", argc, argv)) > 0) order = atoi(argv[i + 1]);
  if ((i = arg_pos((char *)"-negative", argc, argv)) > 0) num_negative = atoi(argv[i + 1]);
  if ((i = arg_pos((char *)"-samples", argc, argv)) > 0) total_samples = atoi(argv[i + 1]);
  if ((i = arg_pos((char *)"-rho", argc, argv)) > 0) init_rho = atof(argv[i + 1]);
  if ((i = arg_pos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  total_samples *= 1000000;
  if (order != 1 && order != 2) {
    printf("Error: order should be eighther 1 or 2!\n");
    exit(1);
  }
  train();
  return 0;
}
