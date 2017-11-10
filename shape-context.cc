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
    
  //The MINSIT dataset
  uci::Database database;
  std::vector<Context> contextVector = std::vector<Context>();
  
  makeImagetteContext(database.input,contextVector);

  // Let's cluster the contexts to disminich dimension

  ContextPrototypes cprototypes;
 
  //  Initialisation
  int j = 0;
  for (int i=0; i<HEIGHT;i++){
    for (int v=0; v<WIDTH; v++){
      if (j<LENGTH/2)
	initProtoContext(cprototypes(i,v), contextVector[j]);
      else
	initProtoContext(cprototypes(i,v), contextVector[LENGTH-j]);
      j++;
    }
  }
  int i = 0;
  j = 0;
  double alpha = 0.1;
  
  // We use the rest of the contexts in contextVector to better our online-kmeans + kohonen maps clustering (off line)
  for (auto it = contextVector.begin()+10;it<contextVector.end()-10;it++){
    winnerProtoContext(cprototypes,(*it),i,j);
    majKohonenContext(i,j,cprototypes,(*it),alpha);
  };


  int nb = 1000;
  int period = 10;
  int frame = 0;
  int periodalpha = 40;
  int q = 0;
  int w = 0;

  for(int h = 0;h<nb;h++){

    

    //  Makes a image every period iterations
    if (h%period==0 && VERBOSE){
      cprototypes.PPM("context",frame++);

    }
    
    //Disminiching the alpha parametter to get a stable cluster
    if (h%periodalpha == 0){
      alpha = alpha*0.95;
      if ((q++)*periodalpha/nb>0.25){
	std::cout << 25*(w++)<< " %"<<std::endl;
	q=0;
      }
    }
    
      
    // Lets get a new written number from MNIST
    database.Next();

    // Storing contexts contextVector
    makeImagetteContext(database.input,contextVector);

    //Better our k-means cluster
    majKohonenContextList(cprototypes,contextVector,alpha);

  }
  //Writing our cluster in a map file
  
  WriteFile("map", cprototypes);
  ContextPrototypes v;
  ReadFile("map", v);
  v.PPM("ici",1);
  return 0;
}
