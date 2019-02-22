#ifndef OPTION_LIST_H
#define OPTION_LIST_H
#include "list.h"

typedef struct{
    char *key;
    char *val;
    int used;
} kvp;


int read_option(char *s, listdn *options);
void option_insert(listdn *l, char *key, char *val);
char *option_find(listdn *l, char *key);
float option_find_float(listdn *l, char *key, float def);
float option_find_float_quiet(listdn *l, char *key, float def);
void option_unused(listdn *l);

#endif
