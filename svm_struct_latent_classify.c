/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_classify.c                                       */
/*                                                                      */
/*   Classification Code for Latent SVM^struct                          */
/*                                                                      */
/*   Author: Chun-Nam Yu                                                */
/*   Date: 9.Nov.08                                                     */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

#include <stdio.h>
#include "svm_struct_latent_api.h"


void read_input_parameters(int argc, char **argv, char *testfile, char *modelfile, char *scorefile, STRUCT_LEARN_PARM *sparm);


int main(int argc, char* argv[]) {
  double avgloss,l;
  long i, correct;

  double score;

  char testfile[1024];
  char modelfile[1024];
	//char labelfile[1024];
  char scorefile[1024];
	//char latentfile[1024];
	//FILE	*flabel;
	FILE *fscore;
  //FILE	*flatent;

  STRUCTMODEL model;
  STRUCT_LEARN_PARM sparm;
  LEARN_PARM lparm;
  KERNEL_PARM kparm;

  SAMPLE testsample;
  LABEL y;
  LATENT_VAR h; 

  /* read input parameters */
  read_input_parameters(argc,argv,testfile,modelfile,scorefile,&sparm);
  //sprintf(labelfile,"%s.labels",filestub);
  //sprintf(scorefile,"%s.scores",filestub);
  //sprintf(latentfile,"%s.latent",filestub);
	//flabel = fopen(labelfile,"w");
  fscore = fopen(scorefile,"w");
	//flatent = fopen(latentfile,"w");

  /* read model file */
  printf("Reading model..."); fflush(stdout);
  model = read_struct_model(modelfile, &sparm);
  printf("done.\n"); 

  /* read test examples */
	printf("Reading test examples..."); fflush(stdout);
  testsample = read_struct_test_examples(testfile,&sparm);
	printf("done.\n");

  init_struct_model(testsample,&model,&sparm,&lparm,&kparm);
  
  avgloss = 0.0;
  correct = 0;
  for (i=0;i<testsample.n;i++) {
    score = classify_struct_example(testsample.examples[i].x,&y,&h,&model,&sparm);
    l = loss(testsample.examples[i].y,y,h,&sparm);
    avgloss += l;
    if (l==0) correct++;

		//print_label(y,flabel);
		//fprintf(flabel,"\n"); fflush(flabel);

    fprintf(fscore,"%lf\n",score);

		//print_latent_var(h,flatent);
		//fprintf(flatent,"\n"); fflush(flatent);

    free_label(y);
    free_latent_var(h); 
  }
	//fclose(flabel);
  fclose(fscore);
	//fclose(flatent);

  printf("Average loss on test set: %.4f\n", avgloss/testsample.n);
  printf("Zero/one error on test set: %.4f\n", 1.0 - ((float) correct)/testsample.n);

  //aseem free_struct_sample(testsample);
  free_struct_model(model,&sparm);

  return(0);

}


void read_input_parameters(int argc, char **argv, char *testfile, char *modelfile, char *scorefile, STRUCT_LEARN_PARM *sparm) {

  long i;
  
  /* set default */
  strcpy(modelfile, "lssvm_model");
  sparm->custom_argc = 0;

  for (i=1;(i<argc)&&((argv[i])[0]=='-');i++) {
    switch ((argv[i])[1]) {
      case '-': strcpy(sparm->custom_argv[sparm->custom_argc++],argv[i]);i++; strcpy(sparm->custom_argv[sparm->custom_argc++],argv[i]);break;  
      default: printf("\nUnrecognized option %s!\n\n",argv[i]); exit(0);    
    }
  }

  if (i>=argc) {
    printf("\nNot enough input parameters!\n\n");
    exit(0);
  }

  strcpy(testfile, argv[i]);
	if(i+1<argc)
  	strcpy(modelfile, argv[i+1]);
	if(i+2<argc)
		strcpy(scorefile,argv[i+2]);

  parse_struct_parameters(sparm);

}
