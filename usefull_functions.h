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
#define BFACTOR 0.7
#define PUT(c) (file.put((char)(c)))

// How indicies are set for an imagette
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



namespace dreco {
  

  typedef uci::Map<WIDTH,HEIGHT,
    uci::Database::imagette::width,
    uci::Database::imagette::height> Prototypes;

  typedef uci::Map<WIDTH,HEIGHT,SECTORS,R> ContextPrototypes;

  typedef uci::Map<WIDTH,HEIGHT,1,LENGTH> FrequencyPrototypes;

  typedef std::vector<std::vector<int>> Context;

   //Forward declarations, see below for details about what theses functions do
  void makeImagetteContext(const uci::Database::imagette& xi, std::vector<Context>& contextVector);
  void clusterImagetteContext(std::vector<Context>& contexts, ContextPrototypes& cprototypes, std::map<int,double>& shapeFrequency);
  void winnerProtoContext(const ContextPrototypes& cprototypes, const Context& xi, int& i, int& j);
  double winningRate(int i_winner, int j_winner, int i, int j,const double alpha);

  // Initialises an imagette prototype
  void initProto(Prototypes::imagette& w,
		 const uci::Database::imagette& xi) {
    for(int i = 0 ; i < uci::Database::imagette::height ; ++i)
      for(int j = 0 ; j < uci::Database::imagette::width ; ++j)
	w(i,j) = (double)(xi(i,j));
  }


  //Initialises a context prototype
  void initProtoContext(ContextPrototypes::imagette& w,
			const Context& xi) {
    for(int i = 0 ; i < R ; ++i)
      for(int j = 0 ; j < SECTORS ; ++j)
	w(i,j) = (double)(xi[i][j]);
  }

  
  //Intitialises a frequency prototype
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

  //Updates an imagette prototype
  void learnProto(double alpha, Prototypes::imagette& w, const uci::Database::imagette& xi){
    for(int i = 0 ; i < uci::Database::imagette::height ; ++i){
      for(int j = 0 ; j < uci::Database::imagette::width ; ++j){
	w(i,j) = w(i,j) + alpha * (xi(i,j)-w(i,j));
      }
    }
  }

  //Updates a context prototype
  void learnProtoContext(double alpha, ContextPrototypes::imagette& w, const Context& xi){
    for(int i = 0 ; i < R ; ++i){
      for(int j = 0 ; j < SECTORS ; ++j){
	w(i,j) = w(i,j) + alpha * (xi[i][j]-w(i,j));
      }
    }
  }

  //Not blurring the edges of the imagette since they contain no info in the MNIST dataset (white)
  uci::Database::imagette blurImagette(uci::Database::imagette& xi){
    uci::Database::imagette yi = xi;
    for(int i = 1 ; i < uci::Database::imagette::height-1 ; ++i){
      for(int j = 1 ; j < uci::Database::imagette::width-1 ; ++j){
	yi(i,j) = (xi(i,j) + BFACTOR*(xi(i+1,j)+xi(i-1,j)+xi(i,j+1)+xi(i,j-1)))/5;
      }
    }
    return yi;
  }
  
  //Will return LENGTH points from the edges of the imagette (Canny filter)
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

  // Calculates the euclydian distante to an imagette prototype
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
  //Creates the context of a  point in an imagette, see https://members.loria.fr/MOBerger/Enseignement/Master2/Exposes/mori-cvpr01.pdf
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

  //Calculates the mean of the distance between all pairs of points in the vector. Used to make the contexts non sensitive to scale of the imagette.
  double meanDist(std::vector<std::pair<int,int>>& v){
    double dist = 0.0;
    for (auto it = v.begin();it<v.end();)
      for(auto iy = it++;iy<v.end();iy++)
	dist+=sqrt((((*it).first-(*iy).first)*((*it).first-(*iy).first))+(((*it).second-(*iy).second)*((*it).second-(*iy).second)));
    dist=2*dist/(v.size()*(v.size()+1));
    return dist;
  }

  //Creates the context of an imagette
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

  //Clusters the contexts of an imagette to reduce the dimension and accelerate the classification process
  void clusterImagetteContext(std::vector<Context>& contexts, ContextPrototypes& cprototypes, std::map<int,double>& shapeFrequency){
    int i=0;int j=0;
    shapeFrequency.clear();
    for (auto it = contexts.begin(); it<contexts.end(); it++){
      winnerProtoContext(cprototypes,*it,i,j);
      shapeFrequency[j + HEIGHT*i]+=10;
   
    }
  }



  //Nearest nieghbour closest imagette prototype
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

  //Update the winner context prototye using k-means+kohonen maps
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

  // Updates the context prototypes according to a list of new contexts (on-line k-means)
void majKohonenContextList(ContextPrototypes& cprototypes, std::vector<Context>& v, double& alpha){
  int i = 0;
  int j = 0;
  for (auto it = v.begin();it<v.end();it++){
    winnerProtoContext(cprototypes,(*it),i,j);
    majKohonenContext(i,j,cprototypes,(*it),alpha);
  }
}
  
  //Displays a context object in the console
  void consoleDisplay(Context& v){
    for(auto it = v.begin();it!=v.end();it++){
      for(auto iy = (*it).begin();iy!=(*it).end();iy++)
	std::cout<<" "<<(*iy);
      std::cout<<" "<<std::endl;
    };
    std::cout<<std::endl;
  }

  //Kohonen incluence function here y=1-0.5*x/alpha
  double funcH(const double x,const double alpha){
    double y = 1.0-0.5*x/alpha;
    if (y>0.0)
      return y;
    return 0; 
  }

  //Calculates the influence on a protoype, depending on the distance from the winner
  double winningRate(int i_winner, int j_winner, int i, int j,const double alpha) {
    double dist = ((i_winner-i)*(i_winner-i)+(j_winner-j)*(j_winner-j));
    double sdist = sqrt(dist);
    double h = funcH(sdist,alpha);
    return h;
  }
  // Used to show the classification choice of the algorithm
  void writeImagette(Prototypes::imagette& xi, int no){
    std::ostringstream os;
    std::ofstream file;
    std::string file_name = "imagette";
    int c = 0;
    os << file_name << '-' 
       << std::setw(6) << std::setfill('0') << no << ".ppm";
    file.open(os.str().c_str());
    if(!file) {
      std::cerr << "Error : uci::Map::PPM : "
		<< "I can't open \"" 
		<< os.str().c_str() 
		<< "\". Closing."
		<< std::endl;
      ::exit(1);
    }
    file << "P6\n" 
	 << uci::Database::imagette::width 
	 << ' ' 
	 << uci::Database::imagette::height
	 << "\n255\n";
    for(int h=0;h<uci::Database::imagette::height;h++)
      {
	for(int ww=0;ww<uci::Database::imagette::width;ww++) {
	  c=255-(unsigned char)(((xi)(h,ww))+.5);
	  PUT(c); PUT(c); PUT(c);
	}
      }
  }

  //Adapted from https://www.stev.org/post/cppreadwritestdmaptoafile

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


//Adapted from https://www.stev.org/post/cppreadwritestdmaptoafile

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

}
