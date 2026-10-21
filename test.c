/* Score token windows with run.c and print the mean NLL, so it can be compared against model.py.
   Usage: test <model.bin> <tokens.bin> <window>
   tokens.bin holds int32 windows of window+1 tokens each; state is reset per window. */
#define TESTING
#include "run.c"

int main(int argc, char *argv[]) {
    if (argc != 4) { fprintf(stderr, "Usage: test <model.bin> <tokens.bin> <window>\n"); return 1; }
    Model model;
    build_model(&model, argv[1]);
    int window = atoi(argv[3]);
    FILE* f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "Couldn't open file %s\n", argv[2]); return 1; }
    int* tokens = malloc((window + 1) * sizeof(int));
    int V = model.config.vocab_size;
    double nll = 0.0;
    long n = 0;
    while (fread(tokens, sizeof(int), window + 1, f) == (size_t)(window + 1)) {
        reset_state(&model);
        for (int t = 0; t < window; t++) {
            float* logits = forward(&model, tokens[t]);
            float max_val = logits[0];
            for (int i = 1; i < V; i++) { if (logits[i] > max_val) max_val = logits[i]; }
            double sum = 0.0;
            for (int i = 0; i < V; i++) { sum += exp(logits[i] - max_val); }
            nll += log(sum) + max_val - logits[tokens[t + 1]];
            n++;
        }
    }
    printf("tokens %ld nll %.6f\n", n, nll / n);
    fclose(f);
    free(tokens);
    free_model(&model);
    return 0;
}
