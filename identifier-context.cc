#include "Base.h"
#include <math.h>
#include <vector>
#include <ostream>
#include <map>
#include <sstream>
#include <cstring>


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

typedef uci::Map<WIDTH,HEIGHT,1,LENGTH> FrequencyPrototypes;

typedef std::vector<std::vector<int>> Context;


// Indicies are defined as follow
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


// Imitilises an imagette prototype
void initProto(Prototypes::imagette& w,
	       const uci::Database::imagette& xi) {
  for(int i = 0 ; i < uci::Database::imagette::height ; ++i)
    for(int j = 0 ; j < uci::Database::imagette::width ; ++j)
      w(i,j) = (double)(xi(i,j));
}

void initProtoContext(ContextPrototypes::imagette& w,
	       const Context& xi) {
  for(int i = 0 ; i < R ; ++i)
    for(int j = 0 ; j < SECTORS ; ++j)
      w(i,j) = (double)(xi[i][j]);
}

//Updates  prototype
void learnProto(double alpha, Prototypes::imagette& w, const uci::Database::imagette& xi){
  for(int i = 0 ; i < uci::Database::imagette::height ; ++i){
    for(int j = 0 ; j < uci::Database::imagette::width ; ++j){
      w(i,j) = w(i,j) + alpha * (xi(i,j)-w(i,j));
    }
  }
}


// Calculates the euclydian distance to a context prototype
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

// Calculates the euclydian distance to a frequency prototype
double distanceProtoFrequency(const FrequencyPrototypes::imagette& w, std::map<int,double>& shapeFrequency)        {
  double distE = 0;
  for(int i = 0 ; i < LENGTH ; ++i){
      double d = w(0,i)-(double)shapeFrequency[i];
      distE += d*d;
}
  return distE;
}


//Calculates nearest neighbour context prototype
void winnerProtoContext(const ContextPrototypes& cprototypes, const Context& xi, int& i, int& j) {
  double minDist = distanceProtoContext(cprototypes(0,0),xi)+1;
  double newDist;
  for(int ii = 0 ; ii < HEIGHT ; ++ii){
    for(int jj = 0 ; jj < WIDTH ; ++jj){
      newDist = distanceProtoContext(cprototypes(ii,jj),xi);
      if (newDist<=minDist){
	minDist = newDist;
	i = ii;
	j = jj;
      }
    }
  }
}
void consoleDisplay(Context& v){
   for(auto it = v.begin();it!=v.end();it++){
     for(auto iy = (*it).begin();iy!=(*it).end();iy++)
       std::cout<<" "<<(*iy);
     std::cout<<" "<<std::endl;
   };
   std::cout<<std::endl;
 }

//Calculates nearest neighbour frequency prototype
void winnerProtoFrequency(const FrequencyPrototypes& fprototypes, std::map<int,double>& shapeFrequency, int& i, int& j) {
  double minDist = distanceProtoFrequency(fprototypes(0,0),shapeFrequency)+1;
  double newDist;
  for(int ii = 0 ; ii < HEIGHT ; ++ii){
    for(int jj = 0 ; jj < WIDTH ; ++jj){
      newDist = distanceProtoFrequency(fprototypes(ii,jj),shapeFrequency);
      if (newDist<minDist){
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
      grad = (-xi(i,j-1)+4*xi(i,j)+xi(i,j+1))*(-xi(i,j-1)+4*xi(i,j)+xi(i,j+1));
      grad+= (xi(i-1,j)+4*xi(i,j)-xi(i+1,j))*(xi(i-1,j)+4*xi(i,j)-xi(i+1,j));
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


void clusterImagetteContext(std::vector<Context>& contexts, ContextPrototypes& cprototypes, std::map<int,double>& shapeFrequency){
  int i=0;int j=0;
  shapeFrequency.clear();
  for (auto it = contexts.begin(); it<contexts.end(); it++){
    winnerProtoContext(cprototypes,*it,i,j);
    shapeFrequency[j + HEIGHT*i]+=10;
   
  }
}

void makeImagetteContext(const uci::Database::imagette& xi, std::vector<Context>& contextVector){
  contextVector.clear();
   auto u = detectEdges(xi);
   auto dist = meanDist(u);
   for (auto it = u.begin();it<u.end();it++){
     auto v = u;
     auto currentPoint = *it;
     v.erase(v.begin()+(it-u.begin()));
     Context cont = createContext(currentPoint,v,dist);
     contextVector.push_back(cont);
   }
 }

void initProtoFreq(FrequencyPrototypes::imagette& w,
	       const uci::Database::imagette& xi,
	        ContextPrototypes& cprototypes) {
  std::map<int,double> shapeFrequency;
  shapeFrequency.clear();
  std::vector<Context> contextVector = std::vector<Context>();
  makeImagetteContext(xi,contextVector);
  clusterImagetteContext(contextVector,cprototypes,shapeFrequency);
  for (int i = 0; i<LENGTH;i++){
    w(0,i) = shapeFrequency[i];
  }
}


//adapted from https://www.stev.org/post/cppreadwritestdmaptoafile

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
    if (h%period==0){
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
