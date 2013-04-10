/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_api.c                                            */
/*                                                                      */
/*   API function definitions for Latent SVM^struct                     */
/*                                                                      */
/*   Author: Chun-Nam Yu                                                */
/*   Date: 17.Dec.08                                                    */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "svm_struct_latent_api_types.h"
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>

#define MAX_INPUT_LINE_LENGTH 10000

void die(const char *message)
{
  if(errno) {
      perror(message); 
  } else {
      printf("ERROR: %s\n", message);
  }
  exit(1);
}

SVECTOR *read_sparse_vector(char *file_name, int object_id, STRUCT_LEARN_PARM *sparm){
    
    int scanned;
    WORD *words = NULL;
    char feature_file[1000];
    sprintf(feature_file, "%s_%d.feature", file_name, object_id);
    FILE *fp = fopen(feature_file, "r");
    
    int length = 0;
    while(!feof(fp)){
        length++;
        words = (WORD *) realloc(words, length*sizeof(WORD));
        if(!words) die("Memory error."); 
        scanned = fscanf(fp, " %d:%f", &words[length-1].wnum, &words[length-1].weight);
        if(scanned < 2) {
            words[length-1].wnum = 0;
            words[length-1].weight = 0.0;
        }
    }
    fclose(fp);

    SVECTOR *fvec = create_svector(words,"",1);
    free(words);

    return fvec;
}

SAMPLE read_struct_test_examples(char *file, STRUCT_LEARN_PARM *sparm) {
/*
  Read input examples {(x_1,y_1),...,(x_n,y_n)} from file.
  The type of pattern x and label y has to follow the definition in 
  svm_struct_latent_api_types.h. Latent variables h can be either
  initialized in this function or by calling init_latent_variables(). 
*/
    SAMPLE sample;

    int i , j;

    // open the file containing candidate bounding box dimensions/labels/featurePath and image label
    FILE *fp = fopen(file, "r");
    if(fp==NULL){
        printf("Error: Cannot open input file %s\n",file);
        exit(1);
    }

    sample.n_pos = 0;
    sample.n_neg = 0;

    fscanf(fp,"%ld", &sample.n);
    sample.examples = (EXAMPLE *) malloc(sample.n*sizeof(EXAMPLE));
    if(!sample.examples) die("Memory error.");

    for(i = 0; i < sample.n; i++){
        fscanf(fp,"%s",sample.examples[i].x.file_name);
        fscanf(fp,"%d",&sample.examples[i].x.n_candidates);

        sample.examples[i].x.boxes = (BBOX *) malloc(sample.examples[i].x.n_candidates*sizeof(BBOX));
        if(!sample.examples[i].x.boxes) die("Memory error.");
        sample.examples[i].x.id_map = (int *) malloc(sample.examples[i].x.n_candidates*sizeof(int));
        if(!sample.examples[i].x.id_map) die("Memory error.");
        sample.examples[i].x.bbox_labels = (int *) malloc (sample.examples[i].x.n_candidates*sizeof(int));
        if(!sample.examples[i].x.bbox_labels) die("Memory error.");
        sample.examples[i].x.phis = (SVECTOR **) malloc(sample.examples[i].x.n_candidates*sizeof(SVECTOR *));
        if(!sample.examples[i].x.phis) die("Memory error.");

        for(j = 0; j < sample.examples[i].x.n_candidates; j++){
            fscanf(fp, "%d", &sample.examples[i].x.boxes[j].min_x);
            fscanf(fp, "%d", &sample.examples[i].x.boxes[j].min_y);
            fscanf(fp, "%d", &sample.examples[i].x.boxes[j].width);
            fscanf(fp, "%d", &sample.examples[i].x.boxes[j].height);
            fscanf(fp, "%d", &sample.examples[i].x.id_map[j]);
            // bbox label can be -1 or 0. For negative images all bbbox labels are 0(meaning negative). 
            // For positive images all bbox labels are -1(meaning unknown). 
            fscanf(fp, "%d", &sample.examples[i].x.bbox_labels[j]);

            sample.examples[i].x.phis[j] = read_sparse_vector(sample.examples[i].x.file_name, sample.examples[i].x.id_map[j], sparm);
        }

        fscanf(fp,"%d",&sample.examples[i].y.label);

        // Image label can be 0(negative image) or 1(positive image)
        if(sample.examples[i].y.label == 0) {
            sample.n_neg++;
            sample.examples[i].x.example_cost = 1.0;
        } else {
            sample.n_pos++;
        }
    }

    for (i = 0; i < sample.n; i++)
    {
        if(sample.examples[i].y.label == 1){
            sample.examples[i].x.example_cost = sparm->weak_weight*((double)sample.n_neg)/((double) sample.n_pos);
        }
    }

    return(sample);
}

SAMPLE read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm) {
/*
  Read input examples {(x_1,y_1),...,(x_n,y_n)} from file.
  The type of pattern x and label y has to follow the definition in 
  svm_struct_latent_api_types.h. Latent variables h can be either
  initialized in this function or by calling init_latent_variables(). 
*/
    SAMPLE sample;

    int i , j; 
    
    // open the file containing candidate bounding box dimensions/labels/featurePath and image label
    FILE *fp = fopen(file, "r");
    if(fp==NULL){
        printf("Error: Cannot open input file %s\n",file);
        exit(1);      
    }

    sample.n_pos = 0;
    sample.n_neg = 0;
    
    fscanf(fp,"%ld", &sample.n);
    
	int is_fs = 0;
	int n_candidates;
	char file_name[1000];
	int n_example = 0;
	int label;
    sample.examples = NULL;
    
    for(i = 0; i < sample.n; i++){        
        fscanf(fp, "%d", &is_fs);
        fscanf(fp, "%d", &label); 
        fscanf(fp, "%s", file_name);
        fscanf(fp, "%d", &n_candidates);
        
        if(!is_fs){
            n_example++;
            sample.examples = (EXAMPLE *) realloc(sample.examples, n_example*sizeof(EXAMPLE));
            if(!sample.examples) die("Memory error.");
            sample.examples[n_example-1].x.supervised_positive = 0;
            
            sample.examples[n_example-1].x.boxes = (BBOX *) malloc(n_candidates*sizeof(BBOX));
            if(!sample.examples[n_example-1].x.boxes) die("Memory error.");
            sample.examples[n_example-1].x.id_map = (int *) malloc(n_candidates*sizeof(int));
            if(!sample.examples[n_example-1].x.id_map) die("Memory error.");
            sample.examples[n_example-1].x.bbox_labels = (int *) malloc (n_candidates*sizeof(int));
            if(!sample.examples[n_example-1].x.bbox_labels) die("Memory error.");
            sample.examples[n_example-1].x.phis = (SVECTOR **) malloc(n_candidates*sizeof(SVECTOR *));
            if(!sample.examples[n_example-1].x.phis) die("Memory error.");
            
            sample.examples[n_example-1].x.n_candidates = n_candidates;
	        strcpy(sample.examples[n_example-1].x.file_name, file_name);
	        
	        for(j = 0; j < n_candidates; j++){
                fscanf(fp, "%d", &sample.examples[n_example-1].x.boxes[j].min_x);
                fscanf(fp, "%d", &sample.examples[n_example-1].x.boxes[j].min_y);
                fscanf(fp, "%d", &sample.examples[n_example-1].x.boxes[j].width);
	            fscanf(fp, "%d", &sample.examples[n_example-1].x.boxes[j].height);
	            fscanf(fp, "%d", &sample.examples[n_example-1].x.id_map[j]);
                fscanf(fp, "%d", &sample.examples[n_example-1].x.bbox_labels[j]);
                sample.examples[n_example-1].x.phis[j] = read_sparse_vector(sample.examples[n_example-1].x.file_name, sample.examples[n_example-1].x.id_map[j], sparm);
            }
            fscanf(fp,"%d",&sample.examples[n_example-1].y.label);
            if(sample.examples[n_example-1].y.label == 0) {
                sample.n_neg++;
                sample.examples[n_example-1].x.example_cost = 1.0;
            } 
            else{
                sample.n_pos++;
            }            
        }
        else{
            for(j = 0; j < n_candidates; j++){
                n_example++;
                sample.examples = (EXAMPLE *) realloc(sample.examples, n_example*sizeof(EXAMPLE));
                if(!sample.examples) die("Memory error.");
                sample.examples[n_example-1].x.supervised_positive = 0;
                
                sample.examples[n_example-1].x.boxes = (BBOX *) malloc(sizeof(BBOX));
                if(!sample.examples[n_example-1].x.boxes) die("Memory error.");
                sample.examples[n_example-1].x.id_map = (int *) malloc(sizeof(int));
                if(!sample.examples[n_example-1].x.id_map) die("Memory error.");
                sample.examples[n_example-1].x.bbox_labels = (int *) malloc (sizeof(int));
                if(!sample.examples[n_example-1].x.bbox_labels) die("Memory error.");
                sample.examples[n_example-1].x.phis = (SVECTOR **) malloc(sizeof(SVECTOR *));
                if(!sample.examples[n_example-1].x.phis) die("Memory error.");   
                
                sample.examples[n_example-1].x.n_candidates = 1;
	            strcpy(sample.examples[n_example-1].x.file_name, file_name);
	            
                fscanf(fp, "%d", &sample.examples[n_example-1].x.boxes[0].min_x);
                fscanf(fp, "%d", &sample.examples[n_example-1].x.boxes[0].min_y);
                fscanf(fp, "%d", &sample.examples[n_example-1].x.boxes[0].width);
	            fscanf(fp, "%d", &sample.examples[n_example-1].x.boxes[0].height);
	            fscanf(fp, "%d", &sample.examples[n_example-1].x.id_map[0]);
	            fscanf(fp, "%d", &sample.examples[n_example-1].x.bbox_labels[0]);
	            sample.examples[n_example-1].x.phis[0] = read_sparse_vector(sample.examples[n_example-1].x.file_name, sample.examples[n_example-1].x.id_map[0], sparm);	
	            sample.examples[n_example-1].y.label = sample.examples[n_example-1].x.bbox_labels[0];
	            
	            if(sample.examples[n_example-1].y.label == 0) {
                    sample.n_neg++;
                    sample.examples[n_example-1].x.example_cost = 1.0;
                } 
                else{
                    sample.n_pos++;
                    sample.examples[n_example-1].x.supervised_positive = 1;
                }
            }
            fscanf(fp,"%d",&label); // just there so that line read from file is complete
        }
        
    }
    sample.n = n_example;
    for (i = 0; i < sample.n; i++)
    {
        if(sample.examples[i].y.label == 1){
            if(sample.examples[i].x.supervised_positive){
                sample.examples[i].x.example_cost = sparm->j*((double)sample.n_neg)/((double) sample.n_pos);
            }
            else{
                sample.examples[i].x.example_cost = sparm->weak_weight*((double)sample.n_neg)/((double) sample.n_pos);
            }            
        }
    }
    return(sample); 
}

/*SAMPLE read_struct_test_examples(char *file, STRUCT_LEARN_PARM *sparm) {

    SAMPLE sample;

    int i , j; 
    
    // open the file containing candidate bounding box dimensions/labels/featurePath and image label
    FILE *fp = fopen(file, "r");
    if(fp==NULL){
        printf("Error: Cannot open input file %s\n",file);
        exit(1);      
    }

    sample.n_pos = 0;
    sample.n_neg = 0;
    
    fscanf(fp,"%ld", &sample.n);
    sample.examples = (EXAMPLE *) malloc(sample.n*sizeof(EXAMPLE));
    if(!sample.examples) die("Memory error.");
    
    int min_x;
	int min_y;
	int width;
	int height;
	int id_map;
	int n_candidates;
	char file_name[1000];
	int bbox_label;
	int n_sup_pos;
    
    for(i = 0; i < sample.n; i++){        
        sample.examples[i].x.supervised_positive = 0;
        n_sup_pos = 0;
        
        fscanf(fp, "%s", file_name);
        fscanf(fp, "%d", &n_candidates);
        
        sample.examples[i].x.boxes = (BBOX *) malloc(n_candidates*sizeof(BBOX));
        if(!sample.examples[i].x.boxes) die("Memory error.");
        sample.examples[i].x.id_map = (int *) malloc(n_candidates*sizeof(int));
        if(!sample.examples[i].x.id_map) die("Memory error.");
        sample.examples[i].x.bbox_labels = (int *) malloc (n_candidates*sizeof(int));
        if(!sample.examples[i].x.bbox_labels) die("Memory error.");
        sample.examples[i].x.phis = (SVECTOR **) malloc(n_candidates*sizeof(SVECTOR *));
        if(!sample.examples[i].x.phis) die("Memory error.");
        
        for(j = 0; j < n_candidates; j++){
            fscanf(fp, "%d", &min_x);
            fscanf(fp, "%d", &min_y);
            fscanf(fp, "%d", &width);
	        fscanf(fp, "%d", &height);
	        fscanf(fp, "%d", &id_map);
            // bbox label can be -1 or 0. For negative images all bbbox labels are 0(meaning negative). 
            // For positive images all bbox labels are -1(meaning unknown). 
            fscanf(fp, "%d", &bbox_label);
            
            if(bbox_label == 1){
                if(n_sup_pos >= 1){
	                sample.n++;
	                sample.examples = (EXAMPLE *) realloc(sample.examples, sample.n*sizeof(EXAMPLE));
                    if(!sample.examples) die("Memory error.");
	            }
	            
	            sample.examples[i+n_sup_pos].x.n_candidates = 1;
	            strcpy(sample.examples[i+n_sup_pos].x.file_name, file_name);

	            sample.examples[i+n_sup_pos].x.boxes = (BBOX *) realloc(sample.examples[i+n_sup_pos].x.boxes, sizeof(BBOX));  
	            sample.examples[i+n_sup_pos].x.boxes[0].min_x = min_x;
	            sample.examples[i+n_sup_pos].x.boxes[0].min_y = min_y;
	            sample.examples[i+n_sup_pos].x.boxes[0].height = height;
	            sample.examples[i+n_sup_pos].x.boxes[0].width = width;
	            
	            sample.examples[i+n_sup_pos].x.id_map = (int *) realloc(sample.examples[i+n_sup_pos].x.id_map, sizeof(int));
	            sample.examples[i+n_sup_pos].x.id_map[0] = id_map;

	            sample.examples[i+n_sup_pos].x.bbox_labels = (int *) realloc (sample.examples[i+n_sup_pos].x.bbox_labels, sizeof(int));
	            sample.examples[i+n_sup_pos].x.bbox_labels[0] = bbox_label;   	                  
	            
	            sample.examples[i+n_sup_pos].x.phis = (SVECTOR **) realloc(sample.examples[i+n_sup_pos].x.phis, sizeof(SVECTOR *));
	            sample.examples[i+n_sup_pos].x.phis[0] = read_sparse_vector(sample.examples[i].x.file_name, sample.examples[i+n_sup_pos].x.id_map[0], sparm);
	            sample.examples[i+n_sup_pos].x.supervised_positive = 1;
	            n_sup_pos++;
	            
	            continue;        	
            }
            if(!sample.examples[i].x.supervised_positive){
	            sample.examples[i].x.n_candidates = n_candidates;
	            strcpy(sample.examples[i].x.file_name, file_name);
	            sample.examples[i].x.boxes[j].min_x = min_x;
	            sample.examples[i].x.boxes[j].min_y = min_y;
	            sample.examples[i].x.boxes[j].width = width;
	            sample.examples[i].x.boxes[j].height = height;
	            sample.examples[i].x.bbox_labels[j] = bbox_label;
		        sample.examples[i].x.id_map[j] = id_map;
                sample.examples[i].x.phis[j] = read_sparse_vector(sample.examples[i].x.file_name, sample.examples[i].x.id_map[j], sparm);	        
	        }
        }

        fscanf(fp,"%d",&sample.examples[i].y.label);
        
        // Image label can be 0(negative image) or 1(positive image)
        if(sample.examples[i].y.label == 0) {
            sample.n_neg++;
            sample.examples[i].x.example_cost = 1.0;
        }
        else if(sample.examples[i].x.supervised_positive){ 
            sample.n_pos += n_sup_pos;
            for(j = 1; j < n_sup_pos; j++){
                sample.examples[i+j].y.label = sample.examples[i].y.label;
            }
	    }         
        else { 
            sample.n_pos++;
        }
        
        if(n_sup_pos){
            i += (n_sup_pos-1);
        }
        
    }
    
    for (i = 0; i < sample.n; i++)
    {
        if(sample.examples[i].y.label == 1){
            if(sample.examples[i].x.supervised_positive){
                sample.examples[i].x.example_cost = ((double)sample.n_neg)/((double) sample.n_pos);
            }
            else{
                sample.examples[i].x.example_cost = sparm->weak_weight*((double)sample.n_neg)/((double) sample.n_pos);
            }
            
        }
    }

    return(sample); 
}*/

void init_struct_model(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, KERNEL_PARM *kparm) {
/*
  Initialize parameters in STRUCTMODEL sm. Set the diminension 
  of the feature space sm->sizePsi. Can also initialize your own
  variables in sm here. 
*/

	sm->n = sample.n;
  sm->sizePsi = sparm->feature_size;
}

void init_latent_variables(SAMPLE *sample, LEARN_PARM *lparm, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Initialize latent variables in the first iteration of training.
  Latent variables are stored at sample.examples[i].h, for 1<=i<=sample.n.
*/

    long i;
    int positive_candidate;

    srand(sparm->rng_seed);
    
    for(i=0;i<sample->n;i++) {
        if(sample->examples[i].y.label == 0) {
            sample->examples[i].h.best_bb = -1;
        } 
        else if(sample->examples[i].y.label == 1) {
            positive_candidate = (int) (((float)sample->examples[i].x.n_candidates)*((float)rand())/(RAND_MAX+1.0));
            sample->examples[i].h.best_bb = positive_candidate;
        } 
    }
}

SVECTOR *psi(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Creates the feature vector \Psi(x,y,h) and return a pointer to 
  sparse vector SVECTOR in SVM^light format. The dimension of the 
  feature vector returned has to agree with the dimension in sm->sizePsi. 
*/
  SVECTOR *fvec=NULL;

  if(y.label == 0){
      WORD *words = words = (WORD *) malloc(sizeof(WORD));
      words[0].wnum = 0;
      words[0].weight = 0.0;
      fvec = create_svector(words,"",1);
      free(words);
  }
  else if(y.label == 1){
      fvec = copy_svector(x.phis[h.best_bb]);
  }

  return(fvec);
}

double classify_struct_example(PATTERN x, LABEL *y, LATENT_VAR *h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Makes prediction with input pattern x with weight vector in sm->w,
  i.e., computing argmax_{(y,h)} <w,psi(x,y,h)>. 
  Output pair (y,h) are stored at location pointed to by 
  pointers *y and *h. 
*/
  h->best_bb = 0;  
  y->label = 0;
	return sprod_ns(sm->w, x.phis[0]); // There isn't any latent var for test images, so we use 0 index        
  //return;
}

void find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, int *robust_candidates) {
/*
  Finds the most violated constraint (loss-augmented inference), i.e.,
  computing argmax_{(ybar,hbar)} [<w,psi(x,ybar,hbar)> + loss(y,ybar,hbar)].
  The output (ybar,hbar) are stored at location pointed by 
  pointers *ybar and *hbar. 
*/
  int i, best_posbb;
  double maxNegScore, posScore;
  double maxPosScore = -DBL_MAX; 
  
  if(y.label == 0){
      // find max negative score, i.e for ybar.label = 0
      maxNegScore = 0;

      // find max positive score, i.e for ybar.label = 1
      for (i = 0; i < x.n_candidates; i++){
          if(robust_candidates[i]){
              posScore = 1 + sprod_ns(sm->w, x.phis[i]);
              if (posScore > maxPosScore)
              {
                maxPosScore = posScore;
                best_posbb = i;
              }
          }
      }
      if (maxPosScore > maxNegScore){
          ybar->label = 1;
          hbar->best_bb = best_posbb;
      }
      else{
          ybar->label = 0;
          hbar->best_bb = -1;
      }
      
  }
  else if(y.label == 1){
      // find max negative score, i.e for ybar.label = 0
      maxNegScore = 1;

      // find max positive score, i.e for ybar.label = 1
      for (i = 0; i < x.n_candidates; i++){
          if (robust_candidates[i]){
              posScore = sprod_ns(sm->w, x.phis[i]);
              if (posScore > maxPosScore)
              {
                maxPosScore = posScore;
                best_posbb = i;
              }
          }
      }
      if (maxPosScore > maxNegScore){
          ybar->label = 1;
          hbar->best_bb = best_posbb;
      }
      else{
          ybar->label = 0;
          hbar->best_bb = -1;
      }
  }

	return;
}

LATENT_VAR infer_latent_variables(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Complete the latent variable h for labeled examples, i.e.,
  computing argmax_{h} <w,psi(x,y,h)>. 
*/

  LATENT_VAR h;

  int i;
  double maxScore = -DBL_MAX;

  if (y.label == 0)
  {
    h.best_bb = -1;
  }
  else{
    maxScore = -DBL_MAX;
    for(i = 0; i < x.n_candidates; i++){
        if(sprod_ns(sm->w, x.phis[i]) > maxScore){
          maxScore = sprod_ns(sm->w, x.phis[i]);
          h.best_bb = i;
        }
    }
  }
  return(h); 
}


double loss(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm) {
/*
  Computes the loss of prediction (ybar,hbar) against the
  correct label y. 
*/ 
	double l;

  if (ybar.label == y.label){
    l = 0;
  }
  else{
    l = 1;
  }

	return(l);
}

void write_struct_model(char *file, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Writes the learned weight vector sm->w to file after training. 
*/
  FILE *modelfl;
  int i;
  
  modelfl = fopen(file,"w");
  if (modelfl==NULL) {
    printf("Cannot open model file %s for output!", file);
		exit(1);
  }
  
  for (i=1;i<sm->sizePsi+1;i++) {
    fprintf(modelfl, "%d:%.16g\n", i, sm->w[i]);
  }
  fclose(modelfl);
 
}

STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm) {
/*
  Reads in the learned model parameters from file into STRUCTMODEL sm.
  The input file format has to agree with the format in write_struct_model().
*/
  STRUCTMODEL sm;

  FILE *modelfl;
  int sizePsi,i, fnum;
  double fweight;
  char line[1000];
  
  modelfl = fopen(file,"r");
  if (modelfl==NULL) {
    printf("Cannot open model file %s for input!", file);
	exit(1);
  }

	sizePsi = 1;
	sm.w = (double*)malloc((sizePsi+1)*sizeof(double));
	for (i=0;i<sizePsi+1;i++) {
		sm.w[i] = 0.0;
	}
	while (!feof(modelfl)) {
		fscanf(modelfl, "%d:%lf", &fnum, &fweight);
		if(fnum > sizePsi) {
			sizePsi = fnum;
			sm.w = (double *)realloc(sm.w,(sizePsi+1)*sizeof(double));
		}
		sm.w[fnum] = fweight;
	}

	fclose(modelfl);

	sm.sizePsi = sizePsi;

  return(sm);

}

void free_struct_model(STRUCTMODEL sm, STRUCT_LEARN_PARM *sparm) {
/*
  Free any memory malloc'ed in STRUCTMODEL sm after training. 
*/

  free(sm.w);

}

void free_pattern(PATTERN x) {
/*
  Free any memory malloc'ed when creating pattern x. 
*/
    int i;

    for(i=0;i<x.n_candidates;i++) {
      free_svector(x.phis[i]);
    }
    if(x.phis != NULL) {
      free(x.phis);
    }
    free(x.boxes);
    free(x.id_map);
}

void free_label(LABEL y) {
/*
  Free any memory malloc'ed when creating label y. 
*/

} 

void free_latent_var(LATENT_VAR h) {
/*
  Free any memory malloc'ed when creating latent variable h. 
*/

}

void free_struct_sample(SAMPLE s) {
/*
  Free the whole training sample. 
*/
  int i;
  for (i=0;i<s.n;i++) {
    free_pattern(s.examples[i].x);
    free_label(s.examples[i].y);
    free_latent_var(s.examples[i].h);
  }
  free(s.examples);

}

void parse_struct_parameters(STRUCT_LEARN_PARM *sparm) {
/*
  Parse parameters for structured output learning passed 
  via the command line. 
*/
  int i;
  
  /* set default */
  sparm->feature_size = 2405;
  sparm->rng_seed = 0;
  sparm->weak_weight = 1e0;
  sparm->robust_cent = 0;
  sparm->j = 1e0;
  
  for (i=0;(i<sparm->custom_argc)&&((sparm->custom_argv[i])[0]=='-');i++) {
    switch ((sparm->custom_argv[i])[2]) {
      /* your code here */
      case 'f': i++; sparm->feature_size = atoi(sparm->custom_argv[i]); break;
      case 'r': i++; sparm->rng_seed = atoi(sparm->custom_argv[i]); break;
      case 'w': i++; sparm->weak_weight = atof(sparm->custom_argv[i]); break;
      case 'p': i++; sparm->robust_cent = atof(sparm->custom_argv[i]); break;
      case 'j': i++; sparm->j = atof(sparm->custom_argv[i]); break;
      default: printf("\nUnrecognized option %s!\n\n", sparm->custom_argv[i]); exit(0);
    }
  }

}

void copy_label(LABEL l1, LABEL *l2)
{
}

void copy_latent_var(LATENT_VAR lv1, LATENT_VAR *lv2)
{
}

void print_latent_var(LATENT_VAR h, FILE *flatent)
{
}

void print_label(LABEL l, FILE	*flabel)
{
}
