#include "darknet_utils.h"
#include <iostream>
char *voc_names[] = {"person",
                     "bicycle",
                     "car",
                     "motorbike",
                     "aeroplane",
                     "bus",
                     "train",
                     "truck",
                     "boat",
                     "traffic light",
                     "fire hydrant",
                     "stop sign",
                     "parking meter",
                     "bench",
                     "bird",
                     "cat",
                     "dog",
                     "horse",
                     "sheep",
                     "cow",
                     "elephant",
                     "bear",
                     "zebra",
                     "giraffe",
                     "backpack",
                     "umbrella",
                     "handbag",
                     "tie",
                     "suitcase",
                     "frisbee",
                     "skis",
                     "snowboard",
                     "sports ball",
                     "kite",
                     "baseball bat",
                     "baseball glove",
                     "skateboard",
                     "surfboard",
                     "tennis racket",
                     "bottle",
                     "wine glass",
                     "cup",
                     "fork",
                     "knife",
                     "spoon",
                     "bowl",
                     "banana",
                     "apple",
                     "sandwich",
                     "orange",
                     "broccoli",
                     "carrot",
                     "hot dog",
                     "pizza",
                     "donut",
                     "cake",
                     "chair",
                     "sofa",
                     "pottedplant",
                     "bed",
                     "diningtable",
                     "toilet",
                     "tvmonitor",
                     "laptop",
                     "mouse",
                     "remote",
                     "keyboard",
                     "cell phone",
                     "microwave",
                     "oven",
                     "toaster",
                     "sink",
                     "refrigerator",
                     "book",
                     "clock",
                     "vase",
                     "scissors",
                     "teddy bear",
                     "hair drier",
                     "toothbrush"};

void average(int argc, char *argv[])
{
    char *cfgfile = argv[2];
    char *outfile = argv[3];
    gpu_index = -1;
    network *net = parse_network_cfg(cfgfile);
    network *sum = parse_network_cfg(cfgfile);

    char *weightfile = argv[4];   
    load_weights(sum, weightfile);

    int i, j;
    int n = argc - 5;
    for(i = 0; i < n; ++i){
        weightfile = argv[i+5];   
        load_weights(net, weightfile);
        for(j = 0; j < net->n; ++j){
            layer l = net->layers[j];
            layer out = sum->layers[j];
            if(l.type == CONVOLUTIONAL){
                int num = l.n*l.c*l.size*l.size;
                axpy_cpu(l.n, 1, l.biases, 1, out.biases, 1);
                axpy_cpu(num, 1, l.weights, 1, out.weights, 1);
                if(l.batch_normalize){
                    axpy_cpu(l.n, 1, l.scales, 1, out.scales, 1);
                    axpy_cpu(l.n, 1, l.rolling_mean, 1, out.rolling_mean, 1);
                    axpy_cpu(l.n, 1, l.rolling_variance, 1, out.rolling_variance, 1);
                }
            }
            if(l.type == CONNECTED){
                axpy_cpu(l.outputs, 1, l.biases, 1, out.biases, 1);
                axpy_cpu(l.outputs*l.inputs, 1, l.weights, 1, out.weights, 1);
            }
        }
    }
    n = n+1;
    for(j = 0; j < net->n; ++j){
        layer l = sum->layers[j];
        if(l.type == CONVOLUTIONAL){
            int num = l.n*l.c*l.size*l.size;
            scal_cpu(l.n, 1./n, l.biases, 1);
            scal_cpu(num, 1./n, l.weights, 1);
                if(l.batch_normalize){
                    scal_cpu(l.n, 1./n, l.scales, 1);
                    scal_cpu(l.n, 1./n, l.rolling_mean, 1);
                    scal_cpu(l.n, 1./n, l.rolling_variance, 1);
                }
        }
        if(l.type == CONNECTED){
            scal_cpu(l.outputs, 1./n, l.biases, 1);
            scal_cpu(l.outputs*l.inputs, 1./n, l.weights, 1);
        }
    }
    save_weights(sum, outfile);
}

long numops(network *net)
{
    int i;
    long ops = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            ops += 2l * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w;
        } else if(l.type == CONNECTED){
            ops += 2l * l.inputs * l.outputs;
        } else if (l.type == RNN){
            ops += 2l * l.input_layer->inputs * l.input_layer->outputs;
            ops += 2l * l.self_layer->inputs * l.self_layer->outputs;
            ops += 2l * l.output_layer->inputs * l.output_layer->outputs;
        } else if (l.type == GRU){
            ops += 2l * l.uz->inputs * l.uz->outputs;
            ops += 2l * l.uh->inputs * l.uh->outputs;
            ops += 2l * l.ur->inputs * l.ur->outputs;
            ops += 2l * l.wz->inputs * l.wz->outputs;
            ops += 2l * l.wh->inputs * l.wh->outputs;
            ops += 2l * l.wr->inputs * l.wr->outputs;
        } else if (l.type == LSTM){
            ops += 2l * l.uf->inputs * l.uf->outputs;
            ops += 2l * l.ui->inputs * l.ui->outputs;
            ops += 2l * l.ug->inputs * l.ug->outputs;
            ops += 2l * l.uo->inputs * l.uo->outputs;
            ops += 2l * l.wf->inputs * l.wf->outputs;
            ops += 2l * l.wi->inputs * l.wi->outputs;
            ops += 2l * l.wg->inputs * l.wg->outputs;
            ops += 2l * l.wo->inputs * l.wo->outputs;
        }
    }
    return ops;
}

void speed(char *cfgfile, int tics)
{
    if (tics == 0) tics = 1000;
    network *net = parse_network_cfg(cfgfile);
    set_batch_network(net, 1);
    int i;
    double time=what_time_is_it_now();
    image im = make_image(net->w, net->h, net->c*net->batch);
    for(i = 0; i < tics; ++i){
        network_predict(net, im.data);
    }
    double t = what_time_is_it_now() - time;
    long ops = numops(net);
    printf("\n%d evals, %f Seconds\n", tics, t);
    printf("Floating Point Operations: %.2f Bn\n", (float)ops/1000000000.);
    printf("FLOPS: %.2f Bn\n", (float)ops/1000000000.*tics/t);
    printf("Speed: %f sec/eval\n", t/tics);
    printf("Speed: %f Hz\n", tics/t);
}

void operations(char *cfgfile)
{
    gpu_index = -1;
    network *net = parse_network_cfg(cfgfile);
    long ops = numops(net);
    printf("Floating Point Operations: %ld\n", ops);
    printf("Floating Point Operations: %.2f Bn\n", (float)ops/1000000000.);
}

void oneoff(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = parse_network_cfg(cfgfile);
    int oldn = net->layers[net->n - 2].n;
    int c = net->layers[net->n - 2].c;
    scal_cpu(oldn*c, .1, net->layers[net->n - 2].weights, 1);
    scal_cpu(oldn, 0, net->layers[net->n - 2].biases, 1);
    net->layers[net->n - 2].n = 11921;
    net->layers[net->n - 2].biases += 5;
    net->layers[net->n - 2].weights += 5*c;
    if(weightfile){
        load_weights(net, weightfile);
    }
    net->layers[net->n - 2].biases -= 5;
    net->layers[net->n - 2].weights -= 5*c;
    net->layers[net->n - 2].n = oldn;
    printf("%d\n", oldn);
    layer l = net->layers[net->n - 2];
    copy_cpu(l.n/3, l.biases, 1, l.biases +   l.n/3, 1);
    copy_cpu(l.n/3, l.biases, 1, l.biases + 2*l.n/3, 1);
    copy_cpu(l.n/3*l.c, l.weights, 1, l.weights +   l.n/3*l.c, 1);
    copy_cpu(l.n/3*l.c, l.weights, 1, l.weights + 2*l.n/3*l.c, 1);
    *net->seen = 0;
    save_weights(net, outfile);
}

void oneoff2(char *cfgfile, char *weightfile, char *outfile, int l)
{
    gpu_index = -1;
    network *net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights_upto(net, weightfile, 0, net->n);
        load_weights_upto(net, weightfile, l, net->n);
    }
    *net->seen = 0;
    save_weights_upto(net, outfile, net->n);
}

void partial(char *cfgfile, char *weightfile, char *outfile, int max)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 1);
    save_weights_upto(net, outfile, max);
}

void print_weights(char *cfgfile, char *weightfile, int n)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 1);
    layer l = net->layers[n];
    int i, j;
    //printf("[");
    for(i = 0; i < l.n; ++i){
        //printf("[");
        for(j = 0; j < l.size*l.size*l.c; ++j){
            //if(j > 0) printf(",");
            printf("%g ", l.weights[i*l.size*l.size*l.c + j]);
        }
        printf("\n");
        //printf("]%s\n", (i == l.n-1)?"":",");
    }
    //printf("]");
}

void rescale_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            rescale_weights(l, 2, -.5);
            break;
        }
    }
    save_weights(net, outfile);
}

void rgbgr_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            rgbgr_weights(l);
            break;
        }
    }
    save_weights(net, outfile);
}

void reset_normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.type == CONVOLUTIONAL && l.batch_normalize) {
            denormalize_convolutional_layer(l);
        }
        if (l.type == CONNECTED && l.batch_normalize) {
            denormalize_connected_layer(l);
        }
        if (l.type == GRU && l.batch_normalize) {
            denormalize_connected_layer(*l.input_z_layer);
            denormalize_connected_layer(*l.input_r_layer);
            denormalize_connected_layer(*l.input_h_layer);
            denormalize_connected_layer(*l.state_z_layer);
            denormalize_connected_layer(*l.state_r_layer);
            denormalize_connected_layer(*l.state_h_layer);
        }
    }
    save_weights(net, outfile);
}

layer normalize_layer(layer l, int n)
{
    int j;
    l.batch_normalize=1;
    l.scales = static_cast<float*>(calloc(n, sizeof(float)));
    for(j = 0; j < n; ++j){
        l.scales[j] = 1;
    }
    l.rolling_mean = static_cast<float*>(calloc(n, sizeof(float)));
    l.rolling_variance = static_cast<float*>(calloc(n, sizeof(float)));
    return l;
}

void normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL && !l.batch_normalize){
            net->layers[i] = normalize_layer(l, l.n);
        }
        if (l.type == CONNECTED && !l.batch_normalize) {
            net->layers[i] = normalize_layer(l, l.outputs);
        }
        if (l.type == GRU && l.batch_normalize) {
            *l.input_z_layer = normalize_layer(*l.input_z_layer, l.input_z_layer->outputs);
            *l.input_r_layer = normalize_layer(*l.input_r_layer, l.input_r_layer->outputs);
            *l.input_h_layer = normalize_layer(*l.input_h_layer, l.input_h_layer->outputs);
            *l.state_z_layer = normalize_layer(*l.state_z_layer, l.state_z_layer->outputs);
            *l.state_r_layer = normalize_layer(*l.state_r_layer, l.state_r_layer->outputs);
            *l.state_h_layer = normalize_layer(*l.state_h_layer, l.state_h_layer->outputs);
            net->layers[i].batch_normalize=1;
        }
    }
    save_weights(net, outfile);
}

void statistics_net(char *cfgfile, char *weightfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.type == CONNECTED && l.batch_normalize) {
            printf("Connected Layer %d\n", i);
            statistics_connected_layer(l);
        }
        if (l.type == GRU && l.batch_normalize) {
            printf("GRU Layer %d\n", i);
            printf("Input Z\n");
            statistics_connected_layer(*l.input_z_layer);
            printf("Input R\n");
            statistics_connected_layer(*l.input_r_layer);
            printf("Input H\n");
            statistics_connected_layer(*l.input_h_layer);
            printf("State Z\n");
            statistics_connected_layer(*l.state_z_layer);
            printf("State R\n");
            statistics_connected_layer(*l.state_r_layer);
            printf("State H\n");
            statistics_connected_layer(*l.state_h_layer);
        }
        printf("\n");
    }
}

void denormalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if ((l.type == DECONVOLUTIONAL || l.type == CONVOLUTIONAL) && l.batch_normalize) {
            denormalize_convolutional_layer(l);
            net->layers[i].batch_normalize=0;
        }
        if (l.type == CONNECTED && l.batch_normalize) {
            denormalize_connected_layer(l);
            net->layers[i].batch_normalize=0;
        }
        if (l.type == GRU && l.batch_normalize) {
            denormalize_connected_layer(*l.input_z_layer);
            denormalize_connected_layer(*l.input_r_layer);
            denormalize_connected_layer(*l.input_h_layer);
            denormalize_connected_layer(*l.state_z_layer);
            denormalize_connected_layer(*l.state_r_layer);
            denormalize_connected_layer(*l.state_h_layer);
            l.input_z_layer->batch_normalize = 0;
            l.input_r_layer->batch_normalize = 0;
            l.input_h_layer->batch_normalize = 0;
            l.state_z_layer->batch_normalize = 0;
            l.state_r_layer->batch_normalize = 0;
            l.state_h_layer->batch_normalize = 0;
            net->layers[i].batch_normalize=0;
        }
    }
    save_weights(net, outfile);
}

void mkimg(char *cfgfile, char *weightfile, int h, int w, int num, char *prefix)
{
    network *net = load_network(cfgfile, weightfile, 0);
    image *ims = get_weights(net->layers[0]);
    int n = net->layers[0].n;
    int z;
    for(z = 0; z < num; ++z){
        image im = make_image(h, w, 3);
        fill_image(im, .5);
        int i;
        for(i = 0; i < 100; ++i){
            image r = copy_image(ims[rand()%n]);
            rotate_image_cw(r, rand()%4);
            random_distort_image(r, 1, 1.5, 1.5);
            int dx = rand()%(w-r.w);
            int dy = rand()%(h-r.h);
            ghost_image(r, im, dx, dy);
            free_image(r);
        }
        char buff[256];
        sprintf(buff, "%s/gen_%d", prefix, z);
        save_image(im, buff);
        free_image(im);
    }
}

void visualize(char *cfgfile, char *weightfile)
{
    network *net = load_network(cfgfile, weightfile, 0);
    visualize_network(net);
}

void test_yolo(char *cfgfile, char *weightfile, char *filename, float thresh)
{
    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    layer l = net->layers[net->n-1];
    set_batch_network(net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    float nms=.4;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net->w, net->h);
        float *X = sized.data;
        time=clock();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));

        int nboxes = 0;
        detection *dets = get_network_boxes(net, 1, 1, thresh, 0, 0, 0, &nboxes);

        if (nms) do_nms_sort(dets, l.side*l.side*l.n, l.classes, nms);

        for(int i=0; i<nboxes; i++)
        {
            detection &d = dets[i];
            int cmax = -1;
            float probmax=-1.0;
            for(int ci=0; ci<d.classes; ci++)
            {
                if(d.prob[ci]>probmax)
                {
                    probmax = d.prob[ci];
                    cmax = ci;
                }
            }
            std::cerr<<d.bbox.x<<" "<<d.bbox.y<<" "<<d.bbox.w<<" "<<d.bbox.h<<" "<<probmax<<" "<<d.objectness<<" "<<voc_names[cmax]<<"\n";
        }

        draw_detections(im, dets, l.side*l.side*l.n, thresh, voc_names, alphabet, 20);
        save_image(im, "predictions");
        show_image(im, "predictions", 0);
        free_detections(dets, nboxes);
        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}
