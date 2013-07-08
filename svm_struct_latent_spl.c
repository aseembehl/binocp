/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_spl.c                                            */
/*                                                                      */
/*   Main Optimization Code for Latent SVM^struct using Self-Paced      */
/*   Learning. NOTE: This implementation modifies the CCCP code by      */
/*   Chun-Nam Yu, specifically the file svm_struct_latent_cccp.c,       */
/*   which is a part of the Latent SVM^struct package available on      */
/*   Chun-Nam Yu's webpage.                                             */
/*                                                                      */
/*   Authors: M. Pawan Kumar and Ben Packer                             */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "svm_struct_latent_api.h"
#include "./svm_light/svm_learn.h"

#define MAX_OUTER_ITER 400

void my_wait_any_key();

void my_read_input_parameters(int argc, char* argv[], char *trainfile, char *modelfile, char *init_modelfile, char *objfile, 
			      LEARN_PARM *learn_parm,STRUCT_LEARN_PARM *struct_parm);

double sprod_nn(double *a, double *b, long n) {
  double ans=0.0;
  long i;
  for (i=1;i<n+1;i++) {
    ans+=a[i]*b[i];
  }
  return(ans);
}

void add_vector_nn(double *w, double *dense_x, long n, double factor) {
  long i;
  for (i=1;i<n+1;i++) {
    w[i]+=factor*dense_x[i];
  }
}

double* add_list_nn(SVECTOR *a, long totwords) 
     /* computes the linear combination of the SVECTOR list weighted
	by the factor of each SVECTOR. assumes that the number of
	features is small compared to the number of elements in the
	list */
{
    SVECTOR *f;
    long i;
    double *sum;

    sum=create_nvector(totwords);

    for(i=0;i<=totwords;i++) 
      sum[i]=0;

    for(f=a;f;f=f->next)  
      add_vector_ns(sum,f,f->factor);

    return(sum);
}

/*int main(int argc, char* argv[]) {

  double *w; // weight vector 
  int outer_iter;
  long m, i, j, k;
  double C_pos, C_neg, epsilon;
  LEARN_PARM learn_parm;
  KERNEL_PARM kernel_parm;
  char trainfile[1024];
  char modelfile[1024];
  char init_modelfile[1024];
  char objfile[1024];
  int MAX_ITER;
  // new struct variables 
  EXAMPLE *ex;
	SAMPLE alldata;
  SAMPLE sample;
	SAMPLE val;
  STRUCT_LEARN_PARM sparm;
  STRUCTMODEL sm;
  
  double decrement;
  double primal_obj, last_primal_obj;
  double stop_crit; 
	char itermodelfile[2000];
	
  // read input parameters 
	my_read_input_parameters(argc, argv, trainfile, modelfile, init_modelfile, objfile, &learn_parm, &sparm); 

  epsilon = learn_parm.eps;
  C_neg = learn_parm.svm_c;
  MAX_ITER = learn_parm.maxiter;

  // read in examples 
  alldata = read_struct_examples(trainfile,&sparm);
  int ntrain = (int) round(1.0*alldata.n); // no validation set 
	
	sample = alldata;
  ex = sample.examples;
  m = sample.n;
  
  // initialization 
  init_struct_model(alldata,&sm,&sparm,&learn_parm,&kernel_parm); 

  w = create_nvector(sm.sizePsi);
  clear_nvector(w, sm.sizePsi);
  
   // added by aseem
  if (sparm.isInitByBinSVM){
    sm = read_struct_model(init_modelfile, &sparm);
    for (i=0;i<sm.sizePsi+1;i++)
      w[i] = sm.w[i]; 
  }// added by aseem

  sm.w = w; // establish link to w, as long as w does not change pointer 

  // some training information 
  printf("C: %.8g\n", C_neg);
  printf("epsilon: %.8g\n", epsilon);
  printf("sample.n: %ld\n", sample.n); 
  printf("sm.sizePsi: %ld\n", sm.sizePsi); fflush(stdout);
  

  // impute latent variable for first iteration 
  //init_latent_variables(&sample,&learn_parm,&sm,&sparm);

  // Impute latent variable using updated weight vector
  outer_iter = 0;
  if (sparm.isInitByBinSVM){
    for (i=0;i<m;i++) {
        free_latent_var(ex[i].h);
        ex[i].h = infer_latent_variables(ex[i].x, ex[i].y, &sm, &sparm, sparm.initIter);
    }
    outer_iter = 1 + sparm.initIter;
  }   
  else{
    init_latent_variables(&sample,&learn_parm,&sm,&sparm);
  }
     
  if (sparm.isInitByBinSVM){
    // add code to compute objective value while resuming
  }
  else{
    last_primal_obj = DBL_MAX;
  }
  decrement = 0;

	// initializations
  double lambda_pos, lambda_neg;    
  int iterations = 0;  
  double eta_pos, eta_neg;
  int r;

  double score;
  int pos_idx;
  int neg_idx;
  double c_loss;
  SVECTOR **fvecs = NULL;

  int n_pos_sample = 1;
  int n_neg_sample = 20;
  int example_per_iter = n_pos_sample + 50*n_neg_sample;
  double norm2;
  double scaleFactor;

  C_pos = C_neg*sample.n_neg*50*sparm.weak_weight/sample.n_pos;
    printf("C: %.8g\n", C_pos); fflush(stdout);

  srand(sparm.seed);
  while ((outer_iter<2)||((!stop_crit)&&(outer_iter<MAX_OUTER_ITER))) { 
    printf("OUTER ITER %d\n", outer_iter); 
    fflush(stdout);

    //solve svm. Compute primal objective
    lambda_pos = 1/(double) C_pos;
    lambda_neg = 1/(double) C_neg;
    iterations = 0;
    while(iterations < MAX_ITER){
        if(iterations % 1000 == 0){
            printf("%d Pegasos iteration\n", iterations); fflush(stdout);
        }

        // learning rate
        eta_pos = 1 / (lambda_pos * (iterations+2));

        // pick a random positive sample
        r = ((int)rand()) % sample.n_pos;
        pos_idx = sample.pos_idx[r]; 

        score = sprod_ns(w, ex[pos_idx].h.phi_h_i);
        c_loss = 1 - ex[pos_idx].y.label*score;
        if (c_loss < 0.0){
          c_loss = 0.0;
        }         
         // scale w 
        scaleFactor = 1.0 - eta_pos*lambda_pos;
        for(k = 1; k < sm.sizePsi+1; k++){
            w[k] = scaleFactor*w[k];
        }
        if (c_loss > 0.0) {
            scaleFactor = eta_pos*ex[pos_idx].y.label/example_per_iter;
            add_vector_ns(w, ex[pos_idx].h.phi_h_i, scaleFactor);
        }
        norm2 = sprod_nn(w, w, sm.sizePsi);
        if (norm2 > 1.0/lambda_pos) {
          scaleFactor = sqrt(1.0/(lambda_pos*norm2));
          for(k = 1; k < sm.sizePsi+1; k++){
            w[k] = scaleFactor*w[k];
          }
        }
        iterations++;

        // pick n_neg_sample random negative sample
        for ( i = 0; i < n_neg_sample; i++){          
          r = ((int)rand()) % sample.n_neg;
          neg_idx = sample.neg_idx[r];
          fvecs = readFeatures(ex[neg_idx].x.file_name, ex[neg_idx].x.n_candidates);
          for (j = 0; j < ex[neg_idx].x.n_candidates; j++){
            eta_neg = 1 / (lambda_neg * (iterations+2)); 
            score = sprod_ns(w, fvecs[j]);
            c_loss = 1 - ex[neg_idx].y.label*score;
            if (c_loss < 0.0){
                c_loss = 0.0;
            }             
             // scale w 
            scaleFactor = 1.0 - eta_neg*lambda_neg;
            for(k = 1; k < sm.sizePsi+1; k++){
                w[k] = scaleFactor*w[k];
            }
            if (c_loss > 0.0) {
                scaleFactor = eta_neg*ex[neg_idx].y.label/example_per_iter;
                add_vector_ns(w, fvecs[j], scaleFactor);
            }
            norm2 = sprod_nn(w, w, sm.sizePsi);
            if (norm2 > 1.0/lambda_neg) {
              scaleFactor = sqrt(1.0/(lambda_neg*norm2));
              for(k = 1; k < sm.sizePsi+1; k++){
                w[k] = scaleFactor*w[k];
              }
            }
            iterations++;
          }
          for (j = 0; j < ex[neg_idx].x.n_candidates; j++){
              free_svector(fvecs[j]);
          }
          free(fvecs);
        } 
    }
    norm2 = sprod_nn(w, w, sm.sizePsi);
    //primal_obj = norm2 * lambda / 2.0;
    primal_obj = norm2 / 2.0;
    int n_examples = 50*sample.n_neg + sample.n_pos;
    for (i=0; i < m; i++) {
      if(i % 200 == 0){
            printf("%ld Loss computation \n", i); fflush(stdout);
      }
      if(ex[i].y.label == 1){
          score = sprod_ns(w, ex[i].h.phi_h_i);
          c_loss = 1 - ex[i].y.label*score;
          if (c_loss < 0.0) c_loss = 0.0;
          primal_obj += C_pos*c_loss/n_examples;
      }
      else{
          fvecs = readFeatures(ex[i].x.file_name, ex[i].x.n_candidates);
          for (j = 0; j < ex[i].x.n_candidates; j++){
              score = sprod_ns(w, fvecs[j]);
              c_loss = 1 - ex[i].y.label*score;
              if (c_loss < 0.0) c_loss = 0.0;
              primal_obj += C_neg*c_loss/n_examples;
          }
          for (j = 0; j < ex[i].x.n_candidates; j++){
              free_svector(fvecs[j]);
          }
          free(fvecs);
      }      
    }
    
    // compute decrement in objective in this outer iteration 
    decrement = last_primal_obj - primal_obj;
    last_primal_obj = primal_obj;
    printf("cccp primal objective: %.4f\n", primal_obj);
		if (outer_iter) {
    	printf("cccp decrement: %.4f\n", decrement); fflush(stdout);
		}
		else {
			printf("cccp decrement: N/A\n"); fflush(stdout);
		}
    
    stop_crit = (decrement<C_neg*epsilon);
  
    // impute latent variable using updated weight vector 
		if(!stop_crit) {
    	for (i=0;i<m;i++) {
        if (ex[i].y.label == 1) {
          free_latent_var(ex[i].h);
          ex[i].h = infer_latent_variables(ex[i].x, ex[i].y, &sm, &sparm, outer_iter);
        }
      }
		}

		sprintf(itermodelfile,"%s.%04d",modelfile,outer_iter);
		write_struct_model(itermodelfile, &sm, &sparm);

    outer_iter++;  
  } // end outer loop
  

  // write structural model 
  write_struct_model(modelfile, &sm, &sparm);
  // skip testing for the moment  
  
  // write objective function value to file	
  FILE *objfl = fopen(objfile, "w");
  if (objfl==NULL) {
    printf("Cannot open model file %s for output!", objfile);
    exit(1);
  }
  fprintf(objfl, "%0.7f\n", last_primal_obj);
  fclose(objfl);

  // free memory 
  free_struct_sample(alldata);
	if(ntrain < alldata.n)
	{
		free(sample.examples);
		free(val.examples);
	}
  free_struct_model(sm, &sparm);
   
  return(0); 
  
}*/

  /*int main(int argc, char* argv[]) {

  double *w; // weight vector 
  double *w_update;
  int outer_iter;
  long m, i, j, k;
  double C, epsilon;
  LEARN_PARM learn_parm;
  KERNEL_PARM kernel_parm;
  char trainfile[1024];
  char modelfile[1024];
  char init_modelfile[1024];
  char objfile[1024];
  int MAX_ITER;
  // new struct variables 
  EXAMPLE *ex;
  SAMPLE alldata;
  SAMPLE sample;
  SAMPLE val;
  STRUCT_LEARN_PARM sparm;
  STRUCTMODEL sm;
  
  double decrement;
  double primal_obj, last_primal_obj;
  double stop_crit; 
  char itermodelfile[2000];
  
  // read input parameters 
  my_read_input_parameters(argc, argv, trainfile, modelfile, init_modelfile, objfile, &learn_parm, &sparm); 

  epsilon = learn_parm.eps;
  C = learn_parm.svm_c;
  MAX_ITER = learn_parm.maxiter;

  // read in examples 
  alldata = read_struct_examples(trainfile,&sparm);
  int ntrain = (int) round(1.0*alldata.n); // no validation set 
  
  sample = alldata;
  ex = sample.examples;
  m = sample.n;
  
  // initialization 
  init_struct_model(alldata,&sm,&sparm,&learn_parm,&kernel_parm); 

  w = create_nvector(sm.sizePsi);
  clear_nvector(w, sm.sizePsi);

  w_update = create_nvector(sm.sizePsi);
  clear_nvector(w_update, sm.sizePsi);
  
   // added by aseem
  if (sparm.isInitByBinSVM){
    sm = read_struct_model(init_modelfile, &sparm);
    for (i=0;i<sm.sizePsi+1;i++)
      w[i] = sm.w[i]; 
  }// added by aseem

  sm.w = w; // establish link to w, as long as w does not change pointer 

  // some training information 
  printf("C: %.8g\n", C);
  printf("epsilon: %.8g\n", epsilon);
  printf("sample.n: %ld\n", sample.n); 
  printf("sm.sizePsi: %ld\n", sm.sizePsi); fflush(stdout);
  

  // impute latent variable for first iteration 
  //init_latent_variables(&sample,&learn_parm,&sm,&sparm);

  // Impute latent variable using updated weight vector
  outer_iter = 0;
  if (sparm.isInitByBinSVM){
    for (i=0;i<m;i++) {
        free_latent_var(ex[i].h);
        ex[i].h = infer_latent_variables(ex[i].x, ex[i].y, &sm, &sparm, sparm.initIter);
    }
    outer_iter = 1 + sparm.initIter;
  }   
  else{
    init_latent_variables(&sample,&learn_parm,&sm,&sparm);
  }
     
  if (sparm.isInitByBinSVM){
    // add code to compute objective value while resuming
  }
  else{
    last_primal_obj = DBL_MAX;
  }
  decrement = 0;

  // initializations 
  double lambda;    
  int iterations = 0;  
  double eta;
  int r;

  double score;
  int pos_idx;
  int neg_idx;
  double c_loss;
  SVECTOR **fvecs = NULL;

  int n_pos_sample = 1;
  int n_neg_sample = 20;
  int example_per_iter = n_pos_sample + 50*n_neg_sample;
  double norm2;
  double scaleFactor;

  srand(sparm.seed);
  while ((outer_iter<2)||((!stop_crit)&&(outer_iter<MAX_OUTER_ITER))) { 

    printf("OUTER ITER %d\n", outer_iter); 
    fflush(stdout);

    clear_nvector(w, sm.sizePsi);

    //solve svm. Compute primal objective
    lambda = 1/(double) C;
    for (iterations = 0; iterations < MAX_ITER; iterations++) {
        if(iterations % 25 == 0){
            printf("%d Pegasos iteration\n", iterations); fflush(stdout);
        }
        // learning rate
        eta = 1 / (lambda * (iterations+2)); 

        // pick a random positive sample
        r = ((int)rand()) % sample.n_pos;
        pos_idx = sample.pos_idx[r];
        score = sprod_ns(w, ex[pos_idx].h.phi_h_i);
        c_loss = 1 - ex[pos_idx].y.label*score;
        if (c_loss < 0.0){
          c_loss = 0.0;
        } 
        if (c_loss > 0.0) {
            scaleFactor = eta*ex[pos_idx].y.label/example_per_iter;
            add_vector_ns(w_update, ex[pos_idx].h.phi_h_i, scaleFactor);
        }

        // pick n_neg_sample random negative sample
        for ( i = 0; i < n_neg_sample; i++){          
          r = ((int)rand()) % sample.n_neg;
          neg_idx = sample.neg_idx[r];
          fvecs = readFeatures(ex[neg_idx].x.file_name, ex[neg_idx].x.n_candidates);
          for (j = 0; j < ex[neg_idx].x.n_candidates; j++){
            score = sprod_ns(w, fvecs[j]);
            c_loss = 1 - ex[neg_idx].y.label*score;
            if (c_loss < 0.0){
                c_loss = 0.0;
            } 
            if (c_loss > 0.0) {
                scaleFactor = eta*ex[neg_idx].y.label/example_per_iter;
                add_vector_ns(w_update, fvecs[j], scaleFactor);
            }
          }
          for (j = 0; j < ex[neg_idx].x.n_candidates; j++){
              free_svector(fvecs[j]);
          }
          free(fvecs);
        } 
        

         // scale w 
        scaleFactor = 1.0 - eta*lambda;
        for(k = 1; k < sm.sizePsi+1; k++){
            w[k] = scaleFactor*w[k];
        }

        add_vector_nn(w, w_update, sm.sizePsi, 1);
        clear_nvector(w_update, sm.sizePsi);

        norm2 = sprod_nn(w, w, sm.sizePsi);
        if (norm2 > 1.0/lambda) {
          scaleFactor = sqrt(1.0/(lambda*norm2));
          for(k = 1; k < sm.sizePsi+1; k++){
            w[k] = scaleFactor*w[k];
          }
        }
    }
    norm2 = sprod_nn(w, w, sm.sizePsi);
    //primal_obj = norm2 * lambda / 2.0;
    primal_obj = norm2 / 2.0;
    int n_examples = 50*sample.n_neg + sample.n_pos;
    for (i=0; i < m; i++) {
      if(i % 500 == 0){
            printf("%ld Loss computation \n", i); fflush(stdout);
      }
      if(ex[i].y.label == 1){
          score = sprod_ns(w, ex[i].h.phi_h_i);
          c_loss = 1 - ex[i].y.label*score;
          if (c_loss < 0.0) c_loss = 0.0;
          primal_obj += C*c_loss/n_examples;
      }
      else{
          fvecs = readFeatures(ex[i].x.file_name, ex[i].x.n_candidates);
          for (j = 0; j < ex[i].x.n_candidates; j++){
              score = sprod_ns(w, fvecs[j]);
              c_loss = 1 - ex[i].y.label*score;
              if (c_loss < 0.0) c_loss = 0.0;
              primal_obj += C*c_loss/n_examples;
          }
          for (j = 0; j < ex[i].x.n_candidates; j++){
              free_svector(fvecs[j]);
          }
          free(fvecs);
      }      
    }
    
    // compute decrement in objective in this outer iteration 
    decrement = last_primal_obj - primal_obj;
    last_primal_obj = primal_obj;
    printf("cccp primal objective: %.4f\n", primal_obj);
    if (outer_iter) {
      printf("cccp decrement: %.4f\n", decrement); fflush(stdout);
    }
    else {
      printf("cccp decrement: N/A\n"); fflush(stdout);
    }
    
    stop_crit = (decrement<C*epsilon);
  
    // impute latent variable using updated weight vector 
    if(!stop_crit) {
      for (i=0;i<m;i++) {
        if (ex[i].y.label == 1) {
          free_latent_var(ex[i].h);
          ex[i].h = infer_latent_variables(ex[i].x, ex[i].y, &sm, &sparm, outer_iter);
        }
      }
    }

    sprintf(itermodelfile,"%s.%04d",modelfile,outer_iter);
    write_struct_model(itermodelfile, &sm, &sparm);

    outer_iter++;  
  } // end outer loop
  

  // write structural model 
  write_struct_model(modelfile, &sm, &sparm);
  // skip testing for the moment  
  
  // write objective function value to file 
  FILE *objfl = fopen(objfile, "w");
  if (objfl==NULL) {
    printf("Cannot open model file %s for output!", objfile);
    exit(1);
  }
  fprintf(objfl, "%0.7f\n", last_primal_obj);
  fclose(objfl);

  // free memory 
  free_struct_sample(alldata);
  if(ntrain < alldata.n)
  {
    free(sample.examples);
    free(val.examples);
  }
  free_struct_model(sm, &sparm);
   
  return(0); 
  
}*/

  int main(int argc, char* argv[]) {

  double *w; // weight vector 
  double *w_update;
  int outer_iter;
  long m, i, j, k;
  double C, epsilon;
  LEARN_PARM learn_parm;
  KERNEL_PARM kernel_parm;
  char trainfile[1024];
  char modelfile[1024];
  char init_modelfile[1024];
  char objfile[1024];
  int MAX_ITER;
  // new struct variables 
  EXAMPLE *ex;
  SAMPLE alldata;
  SAMPLE sample;
  SAMPLE val;
  STRUCT_LEARN_PARM sparm;
  STRUCTMODEL sm;
  
  double decrement;
  double primal_obj, last_primal_obj;
  double stop_crit; 
  char itermodelfile[2000];
  
  // read input parameters 
  my_read_input_parameters(argc, argv, trainfile, modelfile, init_modelfile, objfile, &learn_parm, &sparm); 

  epsilon = learn_parm.eps;
  C = learn_parm.svm_c;
  MAX_ITER = learn_parm.maxiter;

  // read in examples 
  alldata = read_struct_examples(trainfile,&sparm);
  int ntrain = (int) round(1.0*alldata.n); // no validation set 
  
  sample = alldata;
  ex = sample.examples;
  m = sample.n;
  
  // initialization 
  init_struct_model(alldata,&sm,&sparm,&learn_parm,&kernel_parm); 

  w = create_nvector(sm.sizePsi);
  clear_nvector(w, sm.sizePsi);

  w_update = create_nvector(sm.sizePsi);
  clear_nvector(w_update, sm.sizePsi);
  
   // added by aseem
  if (sparm.isInitByBinSVM){
    sm = read_struct_model(init_modelfile, &sparm);
    for (i=0;i<sm.sizePsi+1;i++)
      w[i] = sm.w[i]; 
  }// added by aseem

  sm.w = w; // establish link to w, as long as w does not change pointer 

  // some training information 
  printf("C: %.8g\n", C);
  printf("epsilon: %.8g\n", epsilon);
  printf("sample.n: %ld\n", sample.n); 
  printf("sm.sizePsi: %ld\n", sm.sizePsi); fflush(stdout);
  

  // impute latent variable for first iteration 
  //init_latent_variables(&sample,&learn_parm,&sm,&sparm);

  // Impute latent variable using updated weight vector
  outer_iter = 0;
  if (sparm.isInitByBinSVM){
    for (i=0;i<m;i++) {
        free_latent_var(ex[i].h);
        ex[i].h = infer_latent_variables(ex[i].x, ex[i].y, &sm, &sparm, sparm.initIter);
    }
    outer_iter = 1 + sparm.initIter;
  }   
  else{
    init_latent_variables(&sample,&learn_parm,&sm,&sparm);
  }
     
  if (sparm.isInitByBinSVM){
    // add code to compute objective value while resuming
  }
  else{
    last_primal_obj = DBL_MAX;
  }
  decrement = 0;

  // initializations 
  double lambda;    
  int iterations = 0;  
  double eta;
  int r;

  double score;

  double c_loss, maxc_loss;
  int maxc_lossIndex;
  SVECTOR **fvecs = NULL;

  int n_pos_sample = 1;
  int n_neg_sample = 20;
  int example_per_iter = 1;
  double norm2;
  double scaleFactor;

  srand(sparm.seed);
  while ((outer_iter<2)||((!stop_crit)&&(outer_iter<MAX_OUTER_ITER))) { 

    printf("OUTER ITER %d\n", outer_iter); 
    fflush(stdout);

    clear_nvector(w, sm.sizePsi);

    //solve svm. Compute primal objective
    lambda = 1/(double) C;
    for (iterations = 0; iterations < MAX_ITER; iterations++) {
        if(iterations % 10000 == 0){
            printf("%d Pegasos iteration\n", iterations); fflush(stdout);
        }
        // learning rate
        eta = 1 / (lambda * (iterations+2)); 

        // pick a random positive sample
        r = ((int)rand()) % (sample.n_pos+sample.n_neg);
        if(ex[r].y.label == 1){
          score = sprod_ns(w, ex[r].h.phi_h_i);
          c_loss = 1 - ex[r].y.label*score;
          if (c_loss < 0.0){
            c_loss = 0.0;
          } 
          if (c_loss > 0.0) {
            scaleFactor = eta*ex[r].y.label/example_per_iter;
            add_vector_ns(w_update, ex[r].h.phi_h_i, scaleFactor);
          }
        }
        else{
          fvecs = readFeatures(ex[r].x.file_name, ex[r].x.n_candidates);
          maxc_loss = -DBL_MAX;
          maxc_lossIndex = 0;
          for (j = 0; j < ex[r].x.n_candidates; j++){
            score = sprod_ns(w, fvecs[j]);
            c_loss = 1 - ex[r].y.label*score;
            if(c_loss > maxc_loss){
              maxc_loss = c_loss;
              maxc_lossIndex = j;
            }
          }
          if (maxc_loss < 0.0){
              maxc_loss = 0.0;
          } 
          if (maxc_loss > 0.0) {
              scaleFactor = eta*ex[r].y.label/example_per_iter;
              add_vector_ns(w_update, fvecs[maxc_lossIndex], scaleFactor);
          }
          for (j = 0; j < ex[r].x.n_candidates; j++){
              free_svector(fvecs[j]);
          }
          free(fvecs);
        }

        // scale w 
        scaleFactor = 1.0 - eta*lambda;
        for(k = 1; k < sm.sizePsi+1; k++){
            w[k] = scaleFactor*w[k];
        }

        add_vector_nn(w, w_update, sm.sizePsi, 1);
        clear_nvector(w_update, sm.sizePsi);

        norm2 = sprod_nn(w, w, sm.sizePsi);
        if (norm2 > 1.0/lambda) {
          scaleFactor = sqrt(1.0/(lambda*norm2));
          for(k = 1; k < sm.sizePsi+1; k++){
            w[k] = scaleFactor*w[k];
          }
        }
    }
    norm2 = sprod_nn(w, w, sm.sizePsi);
    //primal_obj = norm2 * lambda / 2.0;
    primal_obj = norm2 / 2.0;
    int n_examples = sample.n_neg + sample.n_pos;
    for (i=0; i < m; i++) {
      if(i % 500 == 0){
            printf("%ld Loss computation \n", i); fflush(stdout);
      }
      if(ex[i].y.label == 1){
          score = sprod_ns(w, ex[i].h.phi_h_i);
          c_loss = 1 - ex[i].y.label*score;
          if (c_loss < 0.0) c_loss = 0.0;
          primal_obj += C*c_loss/n_examples;
      }
      else{
          fvecs = readFeatures(ex[i].x.file_name, ex[i].x.n_candidates);
          maxc_loss = -DBL_MAX;
          for (j = 0; j < ex[i].x.n_candidates; j++){
              score = sprod_ns(w, fvecs[j]);
              c_loss = 1 - ex[i].y.label*score;
              if(c_loss > maxc_loss){
                maxc_loss = c_loss;
              }              
          }
          if (maxc_loss < 0.0) maxc_loss = 0.0;
          primal_obj += C*maxc_loss/n_examples;
          for (j = 0; j < ex[i].x.n_candidates; j++){
              free_svector(fvecs[j]);
          }
          free(fvecs);
      }      
    }
    
    // compute decrement in objective in this outer iteration 
    decrement = last_primal_obj - primal_obj;
    last_primal_obj = primal_obj;
    printf("cccp primal objective: %.4f\n", primal_obj);
    if (outer_iter) {
      printf("cccp decrement: %.4f\n", decrement); fflush(stdout);
    }
    else {
      printf("cccp decrement: N/A\n"); fflush(stdout);
    }
    
    stop_crit = (decrement<C*epsilon);
  
    // impute latent variable using updated weight vector 
    if(!stop_crit) {
      for (i=0;i<m;i++) {
        if (ex[i].y.label == 1) {
          free_latent_var(ex[i].h);
          ex[i].h = infer_latent_variables(ex[i].x, ex[i].y, &sm, &sparm, outer_iter);
        }
      }
    }

    sprintf(itermodelfile,"%s.%04d",modelfile,outer_iter);
    write_struct_model(itermodelfile, &sm, &sparm);

    outer_iter++;  
  } // end outer loop
  

  // write structural model 
  write_struct_model(modelfile, &sm, &sparm);
  // skip testing for the moment  
  
  // write objective function value to file 
  FILE *objfl = fopen(objfile, "w");
  if (objfl==NULL) {
    printf("Cannot open model file %s for output!", objfile);
    exit(1);
  }
  fprintf(objfl, "%0.7f\n", last_primal_obj);
  fclose(objfl);

  // free memory 
  free_struct_sample(alldata);
  if(ntrain < alldata.n)
  {
    free(sample.examples);
    free(val.examples);
  }
  free_struct_model(sm, &sparm);
   
  return(0); 
  
}

void my_read_input_parameters(int argc, char *argv[], char *trainfile, char* modelfile, char *init_modelfile, char *objfile, 
			      LEARN_PARM *learn_parm, STRUCT_LEARN_PARM *struct_parm) {
  
  long i;

  /* set default */
  learn_parm->maxiter=100000;
  learn_parm->svm_c=100.0;
  learn_parm->eps=0.001;
  struct_parm->seed=1;

  struct_parm->custom_argc=0;
  struct_parm->min_cccp_iter=8;
  struct_parm->min_area_ratios[0] = 70; 
  struct_parm->min_area_ratios[1] = 70; 
  struct_parm->min_area_ratios[2] = 65; 
  struct_parm->min_area_ratios[3] = 60; 
  struct_parm->min_area_ratios[4] = 55; 
  struct_parm->min_area_ratios[5] = 50;

  for(i=1;(i<argc) && ((argv[i])[0] == '-');i++) {
    switch ((argv[i])[1]) {
    case 'c': i++; learn_parm->svm_c=atof(argv[i]); break;
    case 'e': i++; learn_parm->eps=atof(argv[i]); break;
    case 'n': i++; learn_parm->maxiter=atol(argv[i]); break;
    case 's': i++; struct_parm->seed=atoi(argv[i]); break;
    case '-': strcpy(struct_parm->custom_argv[struct_parm->custom_argc++],argv[i]);i++; strcpy(struct_parm->custom_argv[struct_parm->custom_argc++],argv[i]);break; 
    default: printf("\nUnrecognized option %s!\n\n",argv[i]);
      exit(0);
    }

  }

  if(i>=argc) {
    printf("\nNot enough input parameters!\n\n");
    my_wait_any_key();
    exit(0);
  }
  strcpy (trainfile, argv[i]);

  if((i+1)<argc) {
    strcpy (modelfile, argv[i+1]);
  }
	else {
		strcpy (modelfile, "lssvm.model");
	}
	strcpy (objfile, argv[i+2]);

  if((i+3)<argc) {
      struct_parm->isInitByBinSVM = 1;
      strcpy (init_modelfile, argv[i+3]);
    }
    else{
      struct_parm->isInitByBinSVM = 0;
    }

    if(struct_parm->isInitByBinSVM){
      struct_parm->initIter = atoi(argv[i+4]);
    }
  
  parse_struct_parameters(struct_parm);
}

void my_wait_any_key()
{
  printf("\n(more)\n");
  (void)getc(stdin);
}
