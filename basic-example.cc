#include "Base.h"
#include <math.h>

// On va stocker les imagettes-prototypes au sein d'une grille.
#define WIDTH  6
#define HEIGHT 5
typedef uci::Map<WIDTH,HEIGHT,
		 uci::Database::imagette::width,
		 uci::Database::imagette::height> Prototypes;

// Pour une imagette, les indices sont definis comme suit
//
//         j
//         |
//   ......|.....
//   ......|.....
//   ......|.....
//   ......|.....
//   ......|.....
//   ......#------ i   img(i,j)
//   ............
//   ............
//   ............
//   ............
//   ............  


// Cette fonction permet d'affecter un prototype (dont les pixels sont
// des double dans [0,255]) a une imagette tiree de la base (dont les
// pixels sont des unsigned char). Le & evite les copies inutiles.
void initProto(Prototypes::imagette& w,
	       const uci::Database::imagette& xi) {
  for(int i = 0 ; i < uci::Database::imagette::height ; ++i)
    for(int j = 0 ; j < uci::Database::imagette::width ; ++j)
      w(i,j) = (double)(xi(i,j));
}

//Mise à jour prototype
void learnProto(double alpha, Prototypes::imagette& w, const uci::Database::imagette& xi){
  for(int i = 0 ; i < uci::Database::imagette::height ; ++i){
    for(int j = 0 ; j < uci::Database::imagette::width ; ++j){
      w(i,j) = w(i,j) + alpha * (xi(i,j)-w(i,j));
    }
  }
}

// Calcul de la distance euclydienne à une imagette prototype
double distanceProto(const Prototypes::imagette& w, const uci::Database::imagette& xi)        {
  double distE = 0;
  for(int i = 0 ; i < uci::Database::imagette::height ; ++i){
    for(int j = 0 ; j < uci::Database::imagette::width ; ++j){
      double d = w(i,j)-(double)xi(i,j);
      distE += d*d;
    }
}
  return distE;
}


//Calcul du prototype le plus proche
void winnerProto(const Prototypes& protos, const uci::Database::imagette& xi, int& i, int& j) {
  double minDist = distanceProto(protos(0,0),xi)+1;
  double newDist;
  for(int ii = 0 ; ii < HEIGHT ; ++ii){
    for(int jj = 0 ; jj < WIDTH ; ++jj){
      newDist = distanceProto(protos(ii,jj),xi);
      if (newDist<=minDist){
	minDist = newDist;
	i = ii;
	j = jj;
      }
    }
  }
}

double funcH(const double x,const double alpha){
  double y = 1.0-0.5*x/alpha;
  if (y>0.0)
    return y;
  return 0; 
}

double winningRate(int i_winner, int j_winner, int i, int j,const double alpha) {
  double dist = ((i_winner-i)*(i_winner-i)+(j_winner-j)*(j_winner-j));
  double sdist = sqrt(dist);
  double h = funcH(sdist,alpha);
  return h;
}


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
    //std::cout << "L'imagette tiree de la base est un " << database.what << std::endl;

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
  for( int u = 0 ; u<5;u++){
      database.Next();
      uci::Database::imagette& xi = database.input; // le & fait que xi est un pointeur, on evite une copie.
      std::cout << "L'imagette tiree de la base est un " << database.what << std::endl;
      winnerProto(prototypes,xi,i,j);
      std::cout << "Je pense " << i <<"," << j << std::endl;
    }
  return 0;
}
