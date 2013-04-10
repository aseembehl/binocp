/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_api_types.h                                      */
/*                                                                      */
/*   API type definitions for Latent SVM^struct                         */
/*                                                                      */
/*   Author: Chun-Nam Yu                                                */
/*   Date: 30.Sep.08                                                    */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

# include "svm_light/svm_common.h"

typedef struct bbScore{
    int img_idx;
    int bb_idx;
    double bb_score;
} BB_SCORE;

typedef struct bbox {
  int min_x;
  int min_y;
  int width;
  int height;
} BBOX;

typedef struct pattern {
  /*
    Type definition for input pattern x
  */
	double example_cost; /* cost of individual example */

  char file_name[1000];
  int n_candidates;
  BBOX *boxes;
  int *id_map;
  SVECTOR **phis;
  int *bbox_labels;
  int supervised_positive;
} PATTERN;

typedef struct label {
  /*
    Type definition for output label y
  */
  int label;
    
} LABEL;

typedef struct _sortStruct {
	  double val;
	  int index;
}  sortStruct;

typedef struct latent_var {
  /*
    Type definition for latent variable h
  */
    int best_bb;
} LATENT_VAR;

typedef struct example {
  PATTERN x;
  LABEL y;
  LATENT_VAR h;
} EXAMPLE;

typedef struct sample {
  long n;
  long n_pos;
  long n_neg;
  EXAMPLE *examples;
} SAMPLE;


typedef struct structmodel {
  double *w;          /* pointer to the learned weights */
  MODEL  *svm_model;  /* the learned SVM model */
  long   sizePsi;     /* maximum number of weights in w */
  /* other information that is needed for the stuctural model can be
     added here, e.g. the grammar rules for NLP parsing */
  long n;             /* number of examples */
} STRUCTMODEL;


typedef struct struct_learn_parm {
  double epsilon;              /* precision for which to solve
				  quadratic program */
  long newconstretrain;        /* number of new constraints to
				  accumulate before recomputing the QP
				  solution */
  double C;                    /* trade-off between margin and loss */
  char   custom_argv[20][1000]; /* string set with the -u command line option */
  int    custom_argc;          /* number of -u command line options */
  int    slack_norm;           /* norm to use in objective function
                                  for slack variables; 1 -> L1-norm, 
				  2 -> L2-norm */
  int    loss_type;            /* selected loss function from -r
				  command line option. Select between
				  slack rescaling (1) and margin
				  rescaling (2) */
  int    loss_function;        /* select between different loss
				  functions via -l command line
				  option */
  /* add your own variables */
  long feature_size;
  int rng_seed;
  double weak_weight;
  double robust_cent;
  double j;

  double gram_regularization;
} STRUCT_LEARN_PARM;

