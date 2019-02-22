#include "darknet.h"
void average(int argc, char *argv[]);
long numops(network *net);
void speed(char *cfgfile, int tics);
void operations(char *cfgfile);
void oneoff(char *cfgfile, char *weightfile, char *outfile);
void oneoff2(char *cfgfile, char *weightfile, char *outfile, int l);
void partial(char *cfgfile, char *weightfile, char *outfile, int max);
void print_weights(char *cfgfile, char *weightfile, int n);
void rescale_net(char *cfgfile, char *weightfile, char *outfile);
void rgbgr_net(char *cfgfile, char *weightfile, char *outfile);
void reset_normalize_net(char *cfgfile, char *weightfile, char *outfile);
layer normalize_layer(layer l, int n);
void normalize_net(char *cfgfile, char *weightfile, char *outfile);
void statistics_net(char *cfgfile, char *weightfile);
void denormalize_net(char *cfgfile, char *weightfile, char *outfile);
void mkimg(char *cfgfile, char *weightfile, int h, int w, int num, char *prefix);
void visualize(char *cfgfile, char *weightfile);
void test_yolo(char *cfgfile, char *weightfile, char *filename, float thresh);
