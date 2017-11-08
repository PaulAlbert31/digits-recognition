#include "Base.h"
#include <math.h>
#include <vector>
#include <ostream>
#include <map>
#include <sstream>
#include <cstring>


#define VERBOSE true
#define WIDTH  6
#define HEIGHT 5
#define LENGTH 40
#define SECTORS 12
#define R 5
#define PUT(c) (file.put((char)(c)))
typedef uci::Map<WIDTH,HEIGHT,
		 uci::Database::imagette::width,
		 uci::Database::imagette::height> Prototypes;

typedef uci::Map<WIDTH,HEIGHT,SECTORS,R> ContextPrototypes;

typedef std::vector<std::vector<int>> Context;


// How indicies are for an imagette
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



void initProtoContext(ContextPrototypes::imagette& w,
	       const Context& xi) {
  for(int i = 0 ; i < R ; ++i)
    for(int j = 0 ; j < SECTORS ; ++j)
      w(i,j) = (double)(xi[i][j]);
}

void learnProtoContext(double alpha, ContextPrototypes::imagette& w, const Context& xi){
  for(int i = 0 ; i < R ; ++i){
    for(int j = 0 ; j < SECTORS ; ++j){
      w(i,j) = w(i,j) + alpha * (xi[i][j]-w(i,j));
    }
  }
}


// Calcul de la distance euclydienne à un context prototype
double distanceProtoContext(const ContextPrototypes::imagette& w, const Context& xi)        {
  double distE = 0;
  for(int i = 0 ; i < R ; ++i){
    for(int j = 0 ; j < SECTORS ; ++j){
      double d = w(i,j)-(double)xi[i][j];
      distE += d*d;
    }
}
  return distE;
}

//Calcul du context prototype le plus proche
void winnerProtoContext(const ContextPrototypes& protos, const Context& xi, int& i, int& j) {
  double minDist = distanceProtoContext(protos(0,0),xi)+1;
  double newDist;
  for(int ii = 0 ; ii < HEIGHT ; ++ii){
    for(int jj = 0 ; jj < WIDTH ; ++jj){
      newDist = distanceProtoContext(protos(ii,jj),xi);
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



//will return LENGTH points from the edges of the imagette (Canny filter)
std::vector<std::pair<int,int>> detectEdges(uci::Database::imagette xi){
  auto yi = std::vector<std::pair<int,int>>();
  double grad;
  for(int i = 1 ; i < uci::Database::imagette::height - 1 ; ++i){
    for(int j = 1 ; j < uci::Database::imagette::width - 1 ; ++j){
      grad = (-xi(i,j-1)+xi(i,j+1))*(-xi(i,j-1)+xi(i,j+1));
      grad+= (xi(i-1,j)-xi(i+1,j))*(xi(i-1,j)-xi(i+1,j));
      grad = sqrt(grad);
      if (grad > 200)
	yi.push_back({i,j});
    }
  }
  auto zi = yi;
  int toDel = yi.size() - LENGTH;
  int deled = 0;
  auto it = zi.begin();
  for(unsigned int i=2;i<yi.size()*2;i=i*2){
    for(unsigned int j = 1;j<i-1;j=j+2){
      if(deled == toDel)
	break;
      zi.erase(it+((int)yi.size()*j/i));
      deled++;
    }
  }
  if(zi.size()!=LENGTH){
    std::cout<<"The number of points in edge detection is too low compared to the LENGTH you chose, consider decreasing it around 30 should be just fine" <<std::endl;;
    return yi;
  }
  return zi;
}

//Create the context of the point see https://members.loria.fr/MOBerger/Enseignement/Master2/Exposes/mori-cvpr01.pdf
Context createContext(const std::pair<int,int>& p, const std::vector<std::pair<int,int>>& v,const  double& alpha){
  double r;
  double teta;
  Context newContext = Context();
  for (int i = 0; i<R;i++)
    newContext.push_back(std::vector<int>(SECTORS,0));    
  for(auto it = v.begin();it<v.end();it++){
     r = sqrt(((*it).first-p.first)*((*it).first-p.first)+((*it).second-p.second)*((*it).second-p.second))/alpha;
     if ((p.second-(*it).second)>0){
       teta = atan((p.first-(*it).first)/(p.second-(*it).second));
       if (teta<0){
	 teta+=6.28;
       }
     }
     else{
       if(p.second == (*it).second){
	 teta = 0.785 ;
       }else{
       teta = atan((p.first-(*it).first)/(p.second-(*it).second))+1.57;
       if (teta<0)
	 teta+=3.14;
       }
     }
     if (r>1){
       newContext[0][((int)(teta/(2*3.14)*SECTORS))%SECTORS] += 10;
     }
     else{
       if(r<(1/pow(2,R-2))){
	 newContext[R-1][((int)(teta/(2*3.14)*SECTORS))%SECTORS] += 10;
       }
       else{
	 newContext[(int)(-log(r)/log(2))+1][((int)(teta/(2*3.14)*SECTORS))%SECTORS] += 10;
       }
      
     }
  }
  return newContext;
}

double meanDist(std::vector<std::pair<int,int>>& v){
  double dist = 0.0;
  for (auto it = v.begin();it<v.end();)
    for(auto iy = it++;iy<v.end();iy++)
      dist+=sqrt((((*it).first-(*iy).first)*((*it).first-(*iy).first))+(((*it).second-(*iy).second)*((*it).second-(*iy).second)));
  dist=2*dist/(v.size()*(v.size()+1));
  return dist;
}


 void consoleDisplay(Context& v){
   for(auto it = v.begin();it!=v.end();it++){
     for(auto iy = (*it).begin();iy!=(*it).end();iy++)
       std::cout<<" "<<(*iy);
     std::cout<<" "<<std::endl;
   };
   std::cout<<std::endl;
 }

void makeImagetteContext(const uci::Database::imagette& xi, std::vector<Context>& contexts){
   auto u = detectEdges(xi);
   auto dist = meanDist(u);
   for (auto it = u.begin();it<u.end();it++){
     auto v = u;
     auto currentPoint = *it;
     v.erase(v.begin()+(it-u.begin()));
     Context cont = createContext(currentPoint,v,dist);
     contexts.push_back(cont);
   }
 }

void majKohonenContext(int& i,int& j, ContextPrototypes& cprototypes, Context& xi, double& alpha){
  for (int k = 0; k< HEIGHT; k++){
      for (int l = 0 ; l< WIDTH;l++){
  	double hh = winningRate(i,j,k,l,alpha);
  	if(hh>0){
  	  for (int m = 0; m< R;m++){
  	    for (int n =0; n<SECTORS;n++){
  	      cprototypes(k,l)(m,n) = cprototypes(k,l)(m,n) + alpha*hh*(xi[m][n]-cprototypes(k,l)(m,n));
  	    }
  	  }
  	}
      }
    }
  }

void majKohonenContextList(ContextPrototypes& cprototypes, std::vector<Context>& v, double& alpha){
  int i = 0;
  int j = 0;
  for (auto it = v.begin();it<v.end();it++){
    winnerProtoContext(cprototypes,(*it),i,j);
    majKohonenContext(i,j,cprototypes,(*it),alpha);
  }
}

//adapted from https://www.stev.org/post/cppreadwritestdmaptoafile

int WriteFile(std::string fname, ContextPrototypes& cprot) {
  int count = 0;
  std::ostringstream strs;
  remove(fname.c_str());
  FILE *fp = fopen(fname.c_str(), "w");
  if (!fp)
    return -errno;
  int u = 0;

  for (int k = 0; k< HEIGHT; k++){
      for (int l = 0 ; l< WIDTH;l++){
  	  for (int m = 0; m< R;m++){
  	    for (int n =0; n<SECTORS;n++){
  	      fprintf(fp, "%s=%s\n",std::to_string((u)).c_str(),std::to_string((int)(cprot(k,l)(m,n))).c_str());
	      count++;
		  
	    }
  	  }
	  u++;
  	}
      }
    

  fclose(fp);
  return count;
}

// Not used here, if you wich to charge an existing cluster to better it, don't forget to set the alpha variable to a low level.
int ReadFile(std::string fname, ContextPrototypes& cprototypes) {
  int count = 0;
  if (access(fname.c_str(), R_OK) < 0)
    return -errno;

  FILE *fp = fopen(fname.c_str(), "r");
  if (!fp)
    return -errno;

  char *buf = 0;
  size_t buflen = 0;
  Context c;
  for (int i = 0; i<R ;){
    c.push_back(std::vector<int>(SECTORS,0));
    i++;
  }

  for (int k = 0; k< HEIGHT; k++){
    for (int l = 0 ; l< WIDTH;l++){
      c.clear();
      for (int u = 0; u< R;u++){
	for (int n =0; n < SECTORS;n++){
	  if(getline(&buf, &buflen, fp)<=0)
	    break;
	  char *nl = strchr(buf, '\n');
	  if (nl == NULL)
	    continue;
	  *nl = 0;

	  char *sep = strchr(buf, '=');
	  if (sep == NULL)
	    continue;
	  *sep = 0;
	  sep++;

	  std::string s1 = buf;
	  std::string s2 = sep;
	  c[u][n] = strtod(sep, NULL);
	  count++;
	}
	}
	initProtoContext(cprototypes(k,l),c);
      }
    }
  

  if (buf)
    free(buf);

  fclose(fp);
  return count;
}











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
