#include "usefull_functions.h"

#define BLUR true
#define BFACTOR 0.7
#define TEST true
#define REDUCENOISE false
#define WIDTH  6
#define HEIGHT 5


int main(int argc, char* argv[]) {

  srand(time(NULL));
  
  Prototypes prototypes;
  
  // Utilisons la base de donnees.
  uci::Database database;

  // Initialisation des 20 premiers prototypes
  for (int i=0; i<HEIGHT;i++){
    for (int v=0; v<WIDTH; v++){
      int v1 = std::rand() % 100;
      for (int j=0;j<v1;j++)
	database.Next();
      initProto(prototypes(i,v), database.input);
    }
  }

  int i = 0;
  int j = 0;
  double alpha = 0.1;
  int nb = 6000;
  int period = 10;
  int frame = 0;
  int periodalpha = 100;
  double halpha = 1.0;
  for(int h = 0;h<nb;h++){

  // On peut afficher la grille de prototypes.
    if (h%period==0){
      prototypes.PPM("kmeans",frame++);

    }
    if (h%periodalpha == 0){
      alpha = alpha*0.95;
      halpha = halpha * 0.99;
    }
      
  // Pour obtenir une nouvelle imagette...
    database.Next();

    uci::Database::imagette& xi = database.input; // le & fait que xi est un pointeur, on evite une copie.
    if(BLUR){
      xi = blurImagette(xi);
    }
    //Mise à jour des prototypes
    winnerProto(prototypes,xi,i,j);
    for (int k = 0; k< HEIGHT; k++){
      for (int l = 0 ; l< WIDTH;l++){
	double hh = winningRate(i,j,k,l,halpha);
	if(hh>0){
	  for (int m = 0; m< uci::Database::imagette::height;m++){
	    for (int n =0; n<uci::Database::imagette::width;n++){
	      prototypes(k,l)(m,n) = prototypes(k,l)(m,n) + alpha*hh*(xi(m,n)-prototypes(k,l)(m,n));

	    }
	  }
	}
      }
    }
  }
  // clearing the noise
  if (REDUCENOISE){
  for (int k = 0; k< HEIGHT; k++){
      for (int l = 0 ; l< WIDTH;l++){
	  for (int m = 0; m< uci::Database::imagette::height;m++){
	    for (int n =0; n<uci::Database::imagette::width;n++){
	      prototypes(k,l)(m,n)  = prototypes(k,l)(m,n)*2;
	      if(prototypes(k,l)(m,n)< 160)
		prototypes(k,l)(m,n) = 0;
	    }
	  }
      }
  }
  prototypes.PPM("result",0);
  }
  if (TEST){
  for( int u = 0 ; u<100;u++){
      database.Next();
      uci::Database::imagette& xi = database.input;
      std::cout << "I pulled a " << database.what << " from the database" << std::endl;
      winnerProto(prototypes,xi,i,j);
      writeImagette(prototypes(i,j),u);
      std::cout << "See imagette-" << u << " for my response" << std::endl;
  }
  }
  return 0;
}
