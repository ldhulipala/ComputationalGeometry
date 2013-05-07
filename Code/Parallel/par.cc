#include <iostream>
using std::cout;
using std::endl;

#include <fstream>
using std::ifstream;

#include <cstdio>
#include <ctime>

#include <cstring>

#include <algorithm> //std::max

#include <vector>
using std::vector;

#include <cstdlib>

#include <climits>

#include <math.h>

#include <mpi.h>

#define MASTER  0

const int MAX_LINE_LN = 512;
const int NUM_ITERS = 20;
const int MAX_TOKENS = 2;
const double DELTA = 0.05;

typedef struct {
  double x, y;
} Point;

typedef struct {
  double xSum, ySum;
  int numPts;
} Centroid;

typedef std::vector<Point*> PointVect;

double dist(double x1, double y1, double x2, double y2) {
  return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

double distance(Point *p, Point *q) {
  return dist(p->x, p->y, q->x, q->y);
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

void printPoints(Point *p, int k) {
  for (int i = 0; i < k; i++) {
    printPoint(&(p[i]));
  }
}

void printCentroid (Centroid *c) {
  printf("Centroid = (%lf, %lf, %d)\n", c->xSum, c->ySum, c->numPts);
}

void printCentroids (Centroid *c, int k) {
   for (int i = 0; i < k; i++) {
     printCentroid(&(c[i]));
   }
}


/* Ensure that k < input.size() */
void initialCenters(PointVect input, Point *writeTo, int k) 
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
    writeTo[i].x = p->x;
    writeTo[i].y = p->y;
  }
}

/* Returns the index of the nearestNeighbour of p in centers */
int nearestNeighbour(Point *p, Point *centers, int k) 
{
  int nn = 0; 
  double curDist = LONG_MAX;
  int clusterNum = 0;
  for (int i = 0; i < k; i++) {
    double dist = LONG_MAX;
    dist = distance(p, &(centers[i]));
    if (dist < curDist) {
      nn = clusterNum;
      curDist = dist;
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

void getCentroid(PointVect cell, Centroid *writeTo) 
{
  if (cell.size() == 0) {
    writeTo->xSum = 0;
    writeTo->ySum = 0;
    writeTo->numPts = 0;
    return;
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
  writeTo->xSum = x;
  writeTo->ySum = y;
  writeTo->numPts = cell.size();
}

void deallocateVect(PointVect v) 
{
  while(!v.empty()) {
    Point *p = v.back();
    v.pop_back();
    delete(p);
  }
}


void sumAndUpdateCenters(Point *centers, PointVect input, Centroid *writeTo, int k) 
{
  PointVect vorCells[k];  
  for (PointVect::iterator it=input.begin(); it != input.end(); ++it)
  {
    int nn = nearestNeighbour(*it, centers, k);
    // add p to the vor-cell of it's NN.
    vorCells[nn].push_back(*it);
  }
  double maxChange = 0;
  for (int i = 0; i < k; i++) {
    Centroid newCentroid;
    getCentroid(vorCells[i], &newCentroid);

    // Save these changes - newCentroid is stack allocated
    writeTo[i].xSum = newCentroid.xSum;
    writeTo[i].ySum = newCentroid.ySum;
    writeTo[i].numPts = newCentroid.numPts;
  }
}

/* Creates a new MPI datatype and writes it to pType */
void createPointType(MPI_Datatype *pType) {
  const int nitems = 2;
  int blocklengths[2] = {1,1};
  MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};
  MPI_Aint offsets[2];

  offsets[0] = offsetof(Point, x);
  offsets[1] = offsetof(Point, y);

  MPI_Type_create_struct(nitems, blocklengths, offsets, types, pType);

  MPI_Type_commit(pType);
}


/* Creates a new MPI datatype and writes it to pType */
void createCentroidType(MPI_Datatype *cType) {
  const int nitems = 3;
  int blocklengths[3] = {1,1,1};
  MPI_Datatype types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
  MPI_Aint offsets[3];

  offsets[0] = offsetof(Centroid, xSum);
  offsets[1] = offsetof(Centroid, ySum);
  offsets[2] = offsetof(Centroid, numPts);

  MPI_Type_create_struct(nitems, blocklengths, offsets, types, cType);

  MPI_Type_commit(cType);
}


int randInt(int i) {
  return std::rand()%i;
}

void computeCenters(Centroid *allC, Point *writeTo, int p, int k, double *d) {
  double drift = 0;
  for (int i = 0; i < k; i++) {
    double x = 0;
    double y = 0;
    int totalPts = 0;
    for (int j = 1; j < p; j++) { // Index from 1 - p(0) is master.
      int at = j*k + i;
      Centroid c = allC[at];
      x += c.xSum;
      y += c.ySum;
      totalPts += c.numPts;
    }
    if (totalPts != 0) {
      x = x / totalPts;
      y = y / totalPts;
    }
    double prevX = writeTo[i].x;
    double prevY = writeTo[i].y;
    writeTo[i].x = x;
    writeTo[i].y = y;
    drift = std::max(drift, dist(prevX, prevY, x, y));
  }
  *d = drift;
}


int main(int argc, char* argv[]) 
{
  std::clock_t start;
  double duration;

  start = std::clock();

  int id;
  int p;

  MPI::Init(argc, argv);
  id = MPI::COMM_WORLD.Get_rank();

  p = MPI::COMM_WORLD.Get_size();

  if (p < 2) {
    printf("Please run the sequential version when using a single processor\n");
    exit(-1);
  }

  /* Create MPI point datatype */
  
  MPI_Datatype point_type;
  createPointType(&point_type);

  MPI_Datatype centroid_type;
  createCentroidType(&centroid_type);

  /* Read in Input Point-Set*/ 
  PointVect inp;
  
  if (argc < 3) {
    printf("Usage : filename, k\n");
    exit(0);
  }

  char *fname = argv[1];
  int k;
  sscanf(argv[2], "%d", &k);

  readInputFile(fname, &inp);

  int numPts = inp.size();
  int avgPts;
  if (inp.size() < (p-1)) {
    avgPts = 1; 
  }
  else {
    avgPts = inp.size() / (p-1);
  }

  int ptArr[numPts];
  int *counts = (int *)malloc(p*sizeof(int));
  int *displs = (int *)malloc(p*sizeof(int));

  for (int i = 0; i < numPts; i++) {
    ptArr[i] = i;
  }

  int remaining = numPts;

  Point processorCents[p][k];

  for (int i = 0; i < p; i++) {
    if (i == 0) {
      /* do not allocate any points to the master */
      counts[i] = 0;
    }

    else if (i == p-1) {
      /* allocate any remaining points to this processor */
      counts[i] = remaining;
      displs[i] = (i-1)*avgPts;
    }

    else {
      /* allocate avgPts many points to this processor */
      counts[i] = avgPts;
      displs[i] = (i-1)*avgPts;
      remaining -= avgPts;
    }
  }

  int toRecv = counts[id];

  int recvbuf[toRecv];
  MPI_Scatterv(&ptArr, counts, displs, MPI_INT, &recvbuf, toRecv, MPI_INT, MASTER,
        MPI_COMM_WORLD);

  /* Now that each slave is assigned some number of pts (possibly 0), filter
   * the input PointVect for points that are assigned to you */

  PointVect nInp;

  int j = 0;
  for (int i=0; i < inp.size(); i++) {
    if (j < toRecv) {
      if (i == recvbuf[j]) {
        Point *p = inp.at(i);
        nInp.push_back(p);
        j += 1; // next pt to take
      }
    }
  }

  Point initC[k];

  if (id == 0) /* Master */
  {
    initialCenters(inp, initC, k);
  }

  /* Broadcast initial centers to all nodes */  
  MPI_Bcast(initC, k, point_type, MASTER, MPI_COMM_WORLD);

  double d = LONG_MAX; // the current global discrepancy, initialy +inf
  int iters = 0; 

/*  Centroid allC[p][k]; */
  Centroid *allC = (Centroid *)malloc(p*k*sizeof(Centroid));
  Centroid nodeC[k];

  // If master, then try to gather all of the nodes centers back to self, 
  // compute some average centers, and then broadcast these centers back to the 
  // nodes. 

  while (d > DELTA && iters < 20) {

    if (id != MASTER) { 
      // Compute the new centers, given current centers are in 
      // initC, nInp is this node's input, and nodeC are the centroids
      // we'll compute for this node.
      sumAndUpdateCenters(initC, nInp, nodeC, k);
    }

    MPI_Gather(nodeC, k, centroid_type, allC, k, centroid_type, MASTER, MPI_COMM_WORLD);


    if (id == MASTER) {
      // Recompute true centers, place them in initC, and broadcast them
      computeCenters(allC,initC, p, k, &d);
    }
    
    MPI_Bcast(initC, k, point_type, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&d, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    iters += 1;
  }

  if (id == MASTER) {
    printf("Iterated %d times. Centers are : \n", iters);
    printPoints(initC, k);
  }

  deallocateVect(inp);

  free(allC);
  free(counts);
  free(displs);

  MPI_Type_free(&point_type);
  MPI_Type_free(&centroid_type);
  MPI_Finalize();

  duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

  cout << "total duration on node " << id << " is " << duration << '\n';

  return 0;
}
