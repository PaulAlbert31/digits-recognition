#include "usefull_functions.h"

#define VERBOSE true
#define WIDTH  6
#define HEIGHT 5
#define LENGTH 40
#define SECTORS 12
#define R 5

using namespace dreco;


int main(int argc, char* argv[]) {

  srand(time(NULL));
  
  Prototypes prototypes;
  
  // Utilisons la base de donnees.
  uci::Database database;
  std::vector<Context> contextVector = std::vector<Context>();
  
  makeImagetteContext(database.input,contextVector);

  // Let's cluster the contexts to disminich dimension

  ContextPrototypes cprototypes;
  FrequencyPrototypes fprototypes;
  // Let's use the cluster we previously created
  ReadFile("map",cprototypes);
  
  

  
  //  First HEIGHT*WIDTH parameters
  for (int i = 0;i<HEIGHT;i++){
    for (int j = 0;j<WIDTH;j++){
      int v1 = std::rand() % 100;
      for (int u=0;u<v1;u++)
	database.Next();
      initProtoFreq(fprototypes(i,j), database.input, cprototypes);
      initProto(prototypes(i,j),database.input);
    }
  }


  int j = 0;
  int i = 0;
  double alpha = 0.9;
  double halpha = 1.0;
  int nb = 3000;
  int period = 10;
  int frame = 0;
  int periodalpha = 40;
  int q = 0;
  int w = 0;
  std::map<int,double> shapeFrequency;

  std::cout<<"Initialisation terminated, starting image clustering"<<std::endl;
  
  for(int h = 0;h<nb;h++){

    

    //Make an image every period iterations
    if (h%period==0 && VERBOSE){
      fprototypes.PPM("frequency",frame++);
      prototypes.PPM("kmeans",frame);

    }
    //Disminiching the alpha parametter to get a stable response
    if (h%periodalpha == 0){
      alpha = alpha*0.95;
      halpha = halpha * 0.99;
      if ((q++)*periodalpha/nb>0.25){
	std::cout << 25*(w++)<< " %"<<std::endl;
	q=0;
      }
    }

    
      
    // Lets go to the next number
    database.Next();

    // Get the image
    auto xi = database.input;

    // Storing contexts contextVector
    makeImagetteContext(xi,contextVector);
    
    // Get the frequency of our clustered contexts
    shapeFrequency.clear();
    clusterImagetteContext(contextVector,cprototypes,shapeFrequency);

    
    // Get the i,j coordinates of the closest frequency prototype
    winnerProtoFrequency(fprototypes,shapeFrequency,i,j);
    

    //Better our k-means + kohonen frequency cluster
    
    for (int k = 0; k< HEIGHT; k++){
      for (int l = 0 ; l< WIDTH;l++){
  	double hh = winningRate(i,j,k,l,halpha);
  	if(hh>0){
	  for (int n = 0; n<LENGTH;n++)
	      fprototypes(k,l)(0,n) = fprototypes(k,l)(0,n) + (double)(alpha*hh*(shapeFrequency[n]-fprototypes(k,l)(0,n)));
	  // to visualise the progression
	  for (int m = 0; m< uci::Database::imagette::height;m++){
	    for (int n =0; n<uci::Database::imagette::width;n++){
	      prototypes(k,l)(m,n) = prototypes(k,l)(m,n) + alpha*hh*(xi(m,n)-prototypes(k,l)(m,n));

	    }
	  }
	}
      }
    }
  }

  return 0;
}
