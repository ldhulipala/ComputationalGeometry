#include <iostream>
using std::cout;
using std::endl;

#include <fstream>
using std::ifstream;

#include <cstring>

#include <algorithm> //std::max

#include <vector>
using std::vector;

#include <cstdlib>

#include <climits>

#include <math.h>

const int MAX_LINE_LN = 512;
const int MAX_TOKENS = 2;
const int NUM_ITERS = 20;
const double DELTA = 0.05;

typedef struct {
  double x, y;
} Point;

typedef std::vector<Point*> PointVect;

double distance(Point *p, Point *q) {
  return sqrt((p->x - q->x) * (p->x - q->x) + (p->y - q->y) * (p->y - q->y));
}

void readInputFile(char *fname, PointVect *v) {

  ifstream fin;
  fin.open(fname);
  if (!fin.good()){
    fin.close();
    return;
  }

  while (!fin.eof() && fin.good()) 
  {
    char buf[MAX_LINE_LN];
    fin.getline(buf, MAX_LINE_LN);

    int i = 0;
    const char* token[MAX_TOKENS] = {};

    token[0] = strtok(buf," ");
    if (token[0])
    {
      for (i=1; i < MAX_TOKENS; i++)
      {
        token[i] = strtok(NULL," ");
        if (!token[i]) break;
      }
      double x;
      double y;
      sscanf(token[0], "%lf", &x);
      sscanf(token[1], "%lf", &y);

      Point *nP = new Point;
      nP->x = x;
      nP->y = y;
      v->push_back(nP);
    }
  }
  fin.close();
}

/* Tests if rd is equal to any of v[0]...v[i-1] */
bool exists_arr(int rd, int i, int v[]){
  for (int j=0; j < i; j++) {
    if (v[j] == rd) {
      return true;
    }
  }
  return false;
}

void printPoint(Point *p) {
  printf("Point = (%lf,%lf)\n",p->x, p->y);
}


/* Ensure that k < input.size() */
void initialCenters(PointVect input, PointVect *writeTo, int k) 
{
  /* First generate k-random centers */
  
  // Can result in different centers per run of the program
//  srand (time(NULL));
  
  srand(1); 

  int result[k]; 
  for (int j = 0; j < k; j++)
  {
    result[j] = -1;
  }
  int n = input.size();
  for (int i = 0; i < k; i++) 
  {
    int rd = rand()%n;
    while (exists_arr(rd,i,result)) {
      rd = rand()%n;
    }
    result[i] = rd;
  }
  /* Allocate NEW points as the 'centers'. We'll continually
   * realloc and dealloc new points as we take each refinement step */
  for (int i = 0; i < k; i++)
  {
    Point *p = input.at(result[i]);
    Point *nP = new Point;
    nP->x = p->x;
    nP->y = p->y;
    writeTo->push_back(nP);
  }
}

/* Returns the index of the nearestNeighbour of p in centers */
int nearestNeighbour(Point *p, PointVect centers) 
{
  int nn = 0; 
  double curDist = LONG_MAX;
  int clusterNum = 0;
  for (PointVect::iterator it=centers.begin(); it != centers.end(); ++it) 
  {
    double dist = LONG_MAX;
    dist = distance(p, *it);
    if (dist < curDist){
      curDist = dist;
      nn = clusterNum;
    }
    clusterNum += 1;
  }
  return nn;
}

void printVect(PointVect v)
{
  for (PointVect::iterator it=v.begin(); it != v.end(); ++it)
  {
    printPoint(*it);
  }
}

Point* getCentroid(PointVect cell) 
{
  if (cell.size() == 0) {
    Point *p = new Point;
    p->x = 0;
    p->y = 0;
    return p;
  }
  double x = 0;
  double y = 0;
  for(PointVect::iterator it=cell.begin(); it != cell.end(); ++it)
  {
    Point *p = *it;
    Point pp = *p;
    x += pp.x;
    y += pp.y;
  }
  x = x/cell.size();
  y = y/cell.size();
  Point *p = new Point;
  p->x = x;
  p->y = y;
  return p;
}

void deallocateVect(PointVect v) 
{
  while(!v.empty()) {
    Point *p = v.back();
    v.pop_back();
    delete(p);
  }
}


void sumAndUpdateCenters(PointVect centers, PointVect input, double* change, PointVect* writeTo) 
{
  int k = centers.size();
  PointVect vorCells[k];  
  for (PointVect::iterator it=input.begin(); it != input.end(); ++it)
  {
    int nn = nearestNeighbour(*it, centers);
    // add p to the vor-cell of it's NN.
    vorCells[nn].push_back(*it);
  }
  double maxChange = 0;
  for (int i = 0; i < k; i++) {
    Point *newCentroid = getCentroid(vorCells[i]);
    Point *oldCentroid = centers.at(i);

    double change = distance(newCentroid, oldCentroid);
    maxChange = std::max(maxChange, change);

    writeTo->push_back(newCentroid);
  }
  *change = maxChange;
}


int main(int argc, char* argv[]) 
{
  /* Set up better wrapppers for robust parsing */
  if (argc < 3) {
    printf("Usage : filename, k\n");
    exit(0);
  }
  char *fname = argv[1];
  int k;
  sscanf(argv[2], "%d", &k);

  PointVect inp;
  readInputFile(fname, &inp);
 
  PointVect *prevCenters = new PointVect;
  initialCenters(inp, prevCenters, k);

  PointVect *newCenters = new PointVect;
  
  double d = LONG_MAX;
  int iters = 0;


  while (d > DELTA && iters < NUM_ITERS) {
    printf("d = %lf\n", d);
    sumAndUpdateCenters(*prevCenters, inp, &d, newCenters);
    deallocateVect(*prevCenters);
    delete(prevCenters);
    prevCenters = newCenters;
    newCenters = new PointVect;
    iters += 1;
  } 
  printf("Finished after %d iterations, centers are : \n", iters);
  printVect(*prevCenters);

  deallocateVect(inp);
  deallocateVect(*prevCenters);
  deallocateVect(*newCenters);
  delete(prevCenters);
  delete(newCenters);

  return 0;
}
