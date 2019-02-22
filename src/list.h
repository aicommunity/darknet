#ifndef LIST_H
#define LIST_H
#include "darknet.h"

listdn *make_list();
int list_find(listdn *l, void *val);

void list_insert(listdn *, void *);


void free_list_contents(listdn *l);

#endif
