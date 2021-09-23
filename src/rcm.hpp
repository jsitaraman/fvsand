void degree ( int root, int adj_row[], int adj[], int mask[], 
  int deg[], int *iccsze, int ls[], int node_num );
void genrcm ( int node_num, int adj_row[], int adj[], int perm[] );
void i4vec_reverse ( int n, int a[] );
void level_set ( int root, int adj_row[], int adj[], int mask[], 
  int *level_num, int level_row[], int level[], int node_num );
void rcm ( int root, int adj_row[], int adj[], int mask[], 
  int perm[], int *iccsze, int node_num );
void root_find ( int *root, int adj_row[], int adj[], int mask[], 
  int *level_num, int level_row[], int level[], int node_num );
