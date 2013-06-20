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

SVECTOR** readFeatures(char *feature_file, int n_fvecs) {
    WORD *words = NULL;
    FILE *fp = fopen(feature_file, "r");
    
    int fvec_length = 0;
    int fvec_buffer_length;
    char *line = NULL;
    size_t len = 0;
    size_t ln;
    char *pair, *single, *brkt, *brkb;

    SVECTOR **fvecs = (SVECTOR **)malloc(n_fvecs*sizeof(SVECTOR *));
    if(!fvecs) die("Memory Error.");

    int i = 0;
    int j = 0;

    /*int offset = 0;
    int wnum;
    float weight;
    char *data;*/

    while( getline(&line,&len,fp) != -1) {
        //data = line;
        ln = strlen(line) - 1;
        if (line[ln] == '\n')
            line[ln] = '\0';
        
        fvec_length = 0;
        fvec_buffer_length = 10000;
        words = (WORD *) malloc(fvec_buffer_length*sizeof(WORD));
        /*while(2 == sscanf(data, " %d:%f%n", &wnum, &weight, &offset)){
            fvec_length++;
            if(fvec_length == fvec_buffer_length){
                fvec_buffer_length = fvec_buffer_length*1.5;
                words = (WORD *) realloc(words, fvec_buffer_length*sizeof(WORD));
            }            
            if(!words) die("Memory error.");
            words[fvec_length-1].wnum = wnum;
            words[fvec_length-1].weight = weight; 
            data += offset;
        }*/
        for(pair = strtok_r(line, " ", &brkt); pair; pair = strtok_r(NULL, " ", &brkt)){
            fvec_length++;
            if(fvec_length == fvec_buffer_length){
                fvec_buffer_length = fvec_buffer_length*1.5;
                words = (WORD *) realloc(words, fvec_buffer_length*sizeof(WORD));
            }            
            if(!words) die("Memory error.");
            j = 0;
            for (single = strtok_r(pair, ":", &brkb); single; single = strtok_r(NULL, ":", &brkb)){
                if(j == 0){
                    words[fvec_length-1].wnum = atoi(single);
                }
                else{
                    words[fvec_length-1].weight = atof(single); 
                }
                j++;
            }
        } 
        fvec_length++; 
        if(fvec_length == fvec_buffer_length){
            words = (WORD *) realloc(words, fvec_length*sizeof(WORD));
            if(!words) die("Memory error.");
        }        
        words[fvec_length-1].wnum = 0;
        words[fvec_length-1].weight = 0.0;

        fvecs[i] = create_svector(words,"",1);
        free(words);
        words = NULL;
        i++;

        free(line);
        line = NULL;
        if (i==n_fvecs){
            break;
        }

   }
   fclose(fp);
   return fvecs;
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
        fscanf(fp,"%d",&sample.examples[i].y.label);
        fscanf(fp,"%d",&sample.examples[i].x.n_candidates);

        sample.examples[i].x.example_cost = 1.0;
        sample.examples[i].y.label == 1;

        sample.examples[i].x.areaRatios = (int *) malloc(sample.examples[i].x.n_candidates*sizeof(int));
        if(!sample.examples[i].x.areaRatios) die("Memory error.");
        for(j = 0; j < sample.examples[i].x.n_candidates; j++){
            sample.examples[i].x.areaRatios[j] = 100;
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
    sample.examples = (EXAMPLE *) malloc(sample.n*sizeof(EXAMPLE));
    if(!sample.examples) die("Memory error.");

    for(i = 0; i < sample.n; i++){
        fscanf(fp,"%s",sample.examples[i].x.file_name);
        fscanf(fp,"%d",&sample.examples[i].y.label);
        fscanf(fp,"%d",&sample.examples[i].x.n_candidates);

        if(sample.examples[i].y.label == 0){
            sample.n_neg++;
            sample.examples[i].x.example_cost = 1.0;
        }
        else {
            sample.n_pos++;
            sample.examples[i].x.areaRatios = (int *) malloc(sample.examples[i].x.n_candidates*sizeof(int));
            if(!sample.examples[i].x.areaRatios) die("Memory error.");

            for(j = 0; j < sample.examples[i].x.n_candidates; j++){
                fscanf(fp, "%d", &sample.examples[i].x.areaRatios[j]);
            }
        }
    }

    for (i = 0; i < sample.n; i++)    {
        if(sample.examples[i].y.label == 1){
            sample.examples[i].x.example_cost = sparm->weak_weight*((double)sample.n_neg)/((double) sample.n_pos);
        }
    }

    return(sample);
}

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

    long i,j;
    int positive_candidate;

    //srand(sparm->rng_seed);

    int maxArea = 0;
    int maxAreaIdx = 0;

    SVECTOR **fvecs;
    
    for(i=0;i<sample->n;i++) {
        if(sample->examples[i].y.label == 0) {
            sample->examples[i].h.best_bb = -1;
        } 
        else if(sample->examples[i].y.label == 1) {
            //positive_candidate = (int) (((float)sample->examples[i].x.n_candidates)*((float)rand())/(RAND_MAX+1.0));
            //sample->examples[i].h.best_bb = positive_candidate;
            maxArea = 0;
            maxAreaIdx = 0;
            for (j = 0; j < sample->examples[i].x.n_candidates; j++)
            {
                if(sample->examples[i].x.areaRatios[j] > maxArea){
                    maxArea = sample->examples[i].x.areaRatios[j];
                    maxAreaIdx = j;
                }
            }
            sample->examples[i].h.best_bb = maxAreaIdx;

            fvecs = readFeatures(sample->examples[i].x.file_name, sample->examples[i].x.n_candidates);
            sample->examples[i].h.phi_h_i = copy_svector(fvecs[sample->examples[i].h.best_bb]);
            for (j = 0; j < sample->examples[i].x.n_candidates; j++){
                free_svector(fvecs[j]);
            }
            free(fvecs);
            if(i % 15 == 0){
                printf("%ld Postive image\n", i); fflush(stdout);
            }
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
      fvec = copy_svector(h.phi_h_i);
  }

  return(fvec);
}



void mine_negative_latent_variables(PATTERN x, LATENT_VAR *hbar, STRUCTMODEL *sm) {
    int i, j;    
    
    double maxScore = -DBL_MAX;
    double score;

    SVECTOR **fvecs = NULL;
    
    fvecs = readFeatures(x.file_name, 50);
    for(j = 0; j < 50; j++){
        score = sprod_ns(sm->w, fvecs[j]);      
        if(score > maxScore){
            maxScore = score;
            hbar->best_bb = j;
        }   
    }
    hbar->phi_h_i = copy_svector(fvecs[hbar->best_bb]);
    for(j =0; j < 50; j++){
        free_svector(fvecs[j]);
    }
    free(fvecs);
}

void find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Finds the most violated constraint (loss-augmented inference), i.e.,
  computing argmax_{(ybar,hbar)} [<w,psi(x,ybar,hbar)> + loss(y,ybar,hbar)].
  The output (ybar,hbar) are stored at location pointed by 
  pointers *ybar and *hbar. 
*/

  double maxNegScore, maxPosScore;

  if(y.label == 0){
      // find max negative score, i.e for ybar.label = 0
      maxNegScore = 0;

      // find max positive score, i.e for ybar.label = 1
      maxPosScore = 1 + sprod_ns(sm->w, hbar->phi_h_i);

      if (maxPosScore > maxNegScore){
          ybar->label = 1;
      }else{
          ybar->label = 0;
      }
  }
  else if(y.label == 1){
      // find max negative score, i.e for ybar.label = 0
      maxNegScore = 1;

      // find max positive score, i.e for ybar.label = 1
      maxPosScore = sprod_ns(sm->w, hbar->phi_h_i);

      if (maxPosScore > maxNegScore){
          ybar->label = 1;
      }else{
          ybar->label = 0;
      }
  }

	return;
}

LATENT_VAR infer_latent_variables(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, int outer_iter) {
/*
  Complete the latent variable h for labeled examples, i.e.,
  computing argmax_{h} <w,psi(x,y,h)>. 
*/

  LATENT_VAR h;

  int j;
  double maxScore = -DBL_MAX;

  SVECTOR **fvecs;

  if (y.label == 0){
      h.best_bb = -1;
      h.phi_h_i = NULL;
  }
  else{
      fvecs = readFeatures(x.file_name, x.n_candidates);
      for(j = 0; j < x.n_candidates; j++){
          if(outer_iter < 6){
              if(x.areaRatios[j] > sparm->min_area_ratios[outer_iter]){
                if(sprod_ns(sm->w, fvecs[j]) > maxScore){
                    maxScore = sprod_ns(sm->w, fvecs[j]);
                    h.best_bb = j;
                }
              }
          }
          else{
              if(sprod_ns(sm->w, fvecs[j]) > maxScore){
                  maxScore = sprod_ns(sm->w, fvecs[j]);
                  h.best_bb = j;
              }
          }
          
      }
      h.phi_h_i = copy_svector(fvecs[h.best_bb]);
      for(j =0; j < x.n_candidates; j++){
          free_svector(fvecs[j]);
      }
      free(fvecs);
  }
  return(h); 
}

double classify_struct_example(PATTERN x, LABEL *y, LATENT_VAR *h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Makes prediction with input pattern x with weight vector in sm->w,
  i.e., computing argmax_{(y,h)} <w,psi(x,y,h)>. 
  Output pair (y,h) are stored at location pointed to by 
  pointers *y and *h. 
*/
  y->label = 1;
  //LATENT_VAR h_classify; 
  *h = infer_latent_variables(x, *y, sm, sparm, 10);
  return sprod_ns(sm->w, h->phi_h_i);        
  //return;
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
    free(x.areaRatios);
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
  free_svector(h.phi_h_i);
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
  sparm->feature_size = 90112;
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
