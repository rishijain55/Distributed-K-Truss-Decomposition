#include "mpi.h"
#include<iostream>
#include <bits/stdc++.h>
#include <time.h>
#include <omp.h>

#define checkIfEdge 1
#define incrementTrussVal 2
#define finishedVal 3
#define commonFlag 0
#define decRem 4
#define decSupp 5
#define setTriangle 6
#define updateEdgeSuppFlag 7
#define updateTrussVal 8
#define updateEdgeTriangle 9
#define allFinishFlag 10
#define ll long long
#define mpill MPI_LONG_LONG

using namespace std;
#define MAXSIZE 100

int *parProc;
vector<int> myNodes;
int curNodeOffset;

struct hash_pair {
    size_t operator()(const std::pair<int, int>& key) const {
        return static_cast<size_t>(key.first) ^ ((static_cast<size_t>(key.second) << 16) | (static_cast<size_t>(key.second) >> 16));
    }
};

unordered_map<pair<int,int>,pair<int,int>, hash_pair> indMap;
 

bool relation(int i, int j, vector<int> &degree){
    if(degree[i] < degree[j]){
        return true;
    }
    else if(degree[i] == degree[j]){
        if(i<j){
            return true;
        }
    }
    return false;
}

int n,m;


class Input{
    vector<string> argv_arguments;
    public:
        Input(int argc, char* argv[]){
            argv_arguments.clear();
            for(int i = 1 ; i < argc ; i ++){
                argv_arguments.push_back(argv[i]);
            }
        }

        bool findInArgv(string s){
            for(int i = 0 ; i < argv_arguments.size() ; i ++){
                if(argv_arguments[i].size() >= s.size() && argv_arguments[i].substr(0, s.size()) == s){
                    return true;
                }
            }
            return false;
        }

        string getArg(string s){
            for(int i = 0 ; i < argv_arguments.size() ; i ++){
                if(argv_arguments[i].size() >= s.size() && argv_arguments[i].substr(0, s.size()) == s){
                    return argv_arguments[i].substr(s.size() + 1);
                }
            }
            return "";
        }

};

void loadHeader(string headerFileName, vector<int> &offsets, vector<int> &degree, FILE* mainFile)
{
    FILE *file = fopen(headerFileName.c_str(), "rb");
    for(int i =0;i<n;i++){
        int offset = getw(file);
        offsets.push_back(offset);
        fseek(mainFile, offsets[i]+4, SEEK_SET);
        int deg;
        deg = getw(mainFile);
        degree.push_back(deg);
    }
    fclose(file);
}


void loadForNode(FILE* file, int node, vector<int> &offsets, vector<int> &adj, vector<int> &degree)
{
    fseek(file, offsets[node]+4, SEEK_SET);
    int deg = getw(file);
    for(int i = 0; i < deg; i++)
    {
        int neighbour;
        neighbour = getw(file);
        if(relation(node, neighbour, degree))
        {
            adj.push_back(neighbour);
        }
    }
}

void loadForNode(FILE* file, int node, vector<int> &offsets, vector<int> &adj, vector<int> &adj2, vector<int> &degree)
{
    fseek(file, offsets[node]+4, SEEK_SET);
    int deg = getw(file);
    for(int i = 0; i < deg; i++)
    {
        int neighbour;
        neighbour = getw(file);
        if(relation(node, neighbour, degree))
        {
            adj.push_back(neighbour);
        }
        adj2.push_back(neighbour);
    }
}


void findCommon(vector<int> &uVector, vector<int> &vVector, int u, int v, vector<int> & ans){
    int i = 0;
    int j = 0;
    while(i < uVector.size() && j < vVector.size()){
        if(uVector[i] == vVector[j]){
            ans.push_back(uVector[i]);
            i++;
            j++;
        }
        else if(uVector[i] < vVector[j]){
            i++;
        }
        else{
            j++;
        }
    }
    
}


void initGraphNew( FILE* file, unordered_map<int,vector<int> > &adj, unordered_map<int,vector<int> > &adj2, vector<int> &offsets, int myRank, int nProcs, vector<int> & degree, int ** &supp,unordered_set<int> ** &triangles)
{
    supp = new int*[myNodes.size()];
    triangles = new unordered_set<int>*[myNodes.size()];
    int p1 =0;
    for(auto i: myNodes){
        vector<int> temp;
        vector<int> temp2;
        fseek(file, offsets[i]+4, SEEK_SET);
        int deg;
        deg = getw(file);
        int p2 =0;
        for(int ind = 0; ind < deg; ind++)
        {
            int neighbour;
            neighbour = getw(file);
            if(relation(i, neighbour, degree))
            {
                temp.push_back(neighbour);
                indMap[{i,neighbour}] = {p1,p2++};
            }
            temp2.push_back(neighbour);
        }
        supp[p1] = new int[temp.size()];
        triangles[p1] = new unordered_set<int>[temp.size()];
        for(int i = 0; i < temp.size(); i++){
            supp[p1][i] = 0;
        }
        p1++;
        adj[i] = temp;
        adj2[i] = temp2;

    }
}


ll suppCalc(int nProcs, int myRank,unordered_map<int,vector<int> > &adj, int **supp, unordered_set<int> **triangles,FILE* file, vector<int> &offsets, vector<int> & degree){
    ll numTri = 0;
    vector<vector<ll> > sendTriples(nProcs, vector<ll>());
    unordered_map<int,vector<int> > otherOne;
    for(auto i: adj){
        for(auto j: i.second){
            otherOne[j].push_back(i.first);
        }
    }
    int it =0;
    for(auto jt:otherOne){

        int curNode = jt.first;
        vector<int> temp;
        if(parProc[curNode]==myRank){
            temp = adj[curNode];
        }
        else{
            loadForNode(file, curNode, offsets, temp, degree);
        }

        if(otherOne.find(curNode) != otherOne.end()){
            for(int jidx = 0 ; jidx < jt.second.size(); jidx++){
                    int j = jt.second[jidx];
                    vector<int> ans;
                    findCommon(temp, adj[j], curNode, j, ans);
                    
                    int u = j;
                    int v = curNode;
                    int recV = parProc[v];
                    if(recV==myRank){
                        for(auto w: ans){
                                pair<int,int> p1 = indMap[{u,v}];
                                pair<int,int> p2 = indMap[{u,w}];
                                pair<int,int> p3 = indMap[{v,w}];
                                supp[p1.first][p1.second]++;
                                supp[p2.first][p2.second]++;
                                supp[p3.first][p3.second]++;
                                triangles[p1.first][p1.second].insert(w);
                                triangles[p2.first][p2.second].insert(v);
                                triangles[p3.first][p3.second].insert(u);
                                numTri+=3;
                        }
                    }
                    else{
                        for(auto w: ans){
                                pair<int,int> p1 = indMap[{u,v}];
                                pair<int,int> p2 = indMap[{u,w}];
                                supp[p1.first][p1.second]++;
                                supp[p2.first][p2.second]++;
                                triangles[p1.first][p1.second].insert(w);
                                triangles[p2.first][p2.second].insert(v);
                                numTri+=2;
                                sendTriples[recV].push_back(v);
                                sendTriples[recV].push_back(w);
                                sendTriples[recV].push_back(u);
                        }
                    }
                    ans.clear();
            }
        }
        temp.clear();
    }
    ll *sendSizes = new ll[nProcs];
    ll *recSizes = new ll[nProcs];
    for(int i = 0 ; i < nProcs ; i ++){
        sendSizes[i] = sendTriples[i].size();
    }

    for(int i = 0 ; i < nProcs ; i ++){
        if(i!=myRank){
            MPI_Request req;
            MPI_Isend(&sendSizes[i], 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &req);
        }
        else{
            recSizes[i] = sendSizes[i];
        }
    }

    for(int i = 0 ; i < nProcs ; i ++){
        if(i!=myRank){
            MPI_Recv(&recSizes[i], 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    ll **recTriples = new ll*[nProcs];
    for(int i = 0 ; i < nProcs ; i ++){
        recTriples[i] = new ll[recSizes[i]];
    }
    for(int i = 0 ; i < nProcs ; i ++){
        if(i!=myRank){
            MPI_Request request;
            MPI_Isend(sendTriples[i].data(), sendSizes[i], MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &request);

        }
        else{
            for(ll j = 0 ; j < sendSizes[i] ; j ++){
                recTriples[i][j] = sendTriples[i][j];
            }
        }
    }
    for(ll i = 0 ; i < nProcs ; i ++){
        if(i!=myRank){
            MPI_Recv(recTriples[i], recSizes[i], MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    for(int i = 0 ; i < nProcs ; i ++){
        for(int j = 0 ; j < recSizes[i] ; j+=3){
            int v = recTriples[i][j];
            int w = recTriples[i][j+1];
            int u = recTriples[i][j+2];
            pair<int,int> p1 = indMap[{v,w}];
            supp[p1.first][p1.second]++;
            triangles[p1.first][p1.second].insert(u);
            numTri++;
        }
    }
    return numTri;
}



int propLight(unordered_map<int,vector<int> > &adj,unordered_map<int,vector<int> > &adjDel,vector<int> &degree, int myRank, int nProcs, int k, int **supp, unordered_set<int> **tri, int &countEdges ){
    int del =1;
    int numIter = 0;
    while(del==1){
        del = 0;
        numIter++;
        vector<int> toDel;
        vector<vector<ll> > triangles(nProcs);
        for(int i : myNodes){
            for(int jind =0;jind<adj[i].size();jind++){
                int j = adj[i][jind];
                pair<int,int> p1 = indMap[{i,j}];
                if( supp[p1.first][p1.second] >=k || adjDel[i][jind]  ) continue;
                adjDel[i][jind] = 1;
                countEdges--;
                del =1;
                for(auto p: tri[p1.first][p1.second]){
                    int w = p;
                    int a2 = i;
                    int b2 = w;
                    int a3 = j;
                    int b3 = w;
                    if(relation(b2, a2, degree)) swap(a2, b2);
                    if(relation(b3, a3, degree)) swap(a3, b3);
                    int reca2 = parProc[a2];
                    int reca3 = parProc[a3];
                    vector<int> temp ={i,j,w};
                    sort(temp.begin(), temp.end());
                    triangles[reca2].push_back(temp[0]);
                    triangles[reca2].push_back(temp[1]);
                    triangles[reca2].push_back(temp[2]);
                    triangles[reca3].push_back(temp[0]);
                    triangles[reca3].push_back(temp[1]);
                    triangles[reca3].push_back(temp[2]);
                }
            }
        }
        ll* recSizes = new ll[nProcs];

        ll* sendSizes = new ll[nProcs];
        for(int i = 0 ; i < nProcs ; i++){
            sendSizes[i] = triangles[i].size();
        }
        for(int i = 0 ; i < nProcs ; i++){
            if(i!=myRank){
                MPI_Request req;
                MPI_Isend(sendSizes+i, 1, MPI_LONG_LONG, i, commonFlag, MPI_COMM_WORLD, &req);
            }
            else{
                recSizes[i] = sendSizes[i];
                if(recSizes[i]>0 ) del = 1;
            }
        }
        for(int i = 0 ; i < nProcs ; i++){
            if(i!=myRank){
                MPI_Recv(recSizes+i, 1, MPI_LONG_LONG, i, commonFlag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if(recSizes[i]>0 ) del = 1;
            }
        }

        ll* recTrianglesrec[nProcs];
        for(int i = 0 ; i < nProcs ; i++){
            recTrianglesrec[i] = new ll[recSizes[i]];
        }
        for(int i = 0 ; i < nProcs ; i++){
            if(i!=myRank){
                MPI_Request req;
                MPI_Isend(triangles[i].data(), sendSizes[i], MPI_LONG_LONG, i, commonFlag, MPI_COMM_WORLD, &req);
            }
            else{
                for(ll j = 0 ; j < recSizes[i] ; j++){
                    recTrianglesrec[i][j] = triangles[i][j];
                }
            }
        }
        for(int i = 0 ; i < nProcs ; i++){
            if(i!=myRank){
                MPI_Recv(recTrianglesrec[i], recSizes[i], MPI_LONG_LONG, i, commonFlag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        for(int i = 0 ; i < nProcs ; i++){
            for(int j = 0 ; j < recSizes[i] ; j+=3){
                // #pragma omp task
                {
                    int a = recTrianglesrec[i][j];
                    int b = recTrianglesrec[i][j+1];
                    int c = recTrianglesrec[i][j+2];
                    int recA = parProc[a];
                    int recB = parProc[b];
                    int recC = parProc[c];
                    if(recA==myRank){
                        // #pragma omp critical
                        {
                            pair<int,int> p1 = indMap[{a,b}];
                            if(relation(a, b, degree) && tri[p1.first][p1.second].find(c)!=tri[p1.first][p1.second].end()){
                                supp[p1.first][p1.second]--;
                                tri[p1.first][p1.second].erase(c);
                            }
                            p1 = indMap[{a,c}];
                            if(relation(a, c, degree) && tri[p1.first][p1.second].find(b)!=tri[p1.first][p1.second].end()) {
                                supp[p1.first][p1.second]--;
                                tri[p1.first][p1.second].erase(b);
                            }
                        }
                    }
                    if(recB==myRank){
                        {
                            pair<int,int> p1 = indMap[{b,a}];
                            if(relation(b, a, degree) && tri[p1.first][p1.second].find(c)!=tri[p1.first][p1.second].end()){
                                supp[p1.first][p1.second]--;
                                tri[p1.first][p1.second].erase(c);
                            }
                            p1 = indMap[{b,c}];
                            if(relation(b, c, degree) && tri[p1.first][p1.second].find(a)!=tri[p1.first][p1.second].end()) {
                                supp[p1.first][p1.second]--;
                                tri[p1.first][p1.second].erase(a);
                            }
                        }
                    }
                    if(recC==myRank){
                        // #pragma omp critical
                        {
                            pair<int,int> p1 = indMap[{c,a}];
                            if(relation(c, a, degree) && tri[p1.first][p1.second].find(b)!=tri[p1.first][p1.second].end()){
                                supp[p1.first][p1.second]--;
                                tri[p1.first][p1.second].erase(b);
                            }
                            p1 = indMap[{c,b}];
                            if(relation(c, b, degree) && tri[p1.first][p1.second].find(a)!=tri[p1.first][p1.second].end()) {
                                supp[p1.first][p1.second]--;
                                tri[p1.first][p1.second].erase(a);
                            }
                        }
                    }
                }
            }
        }
        int newDel = 0;
        MPI_Allreduce(&del, &newDel, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        del = newDel;

    }
    int fl =0;
    if(countEdges>0) fl = 1;
    int newFl;
    MPI_Allreduce(&fl, &newFl, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    fl = newFl;
    return  fl;
}

void bfs(unordered_map<int, vector<int> > &adj, int nProcs, int myRank,unordered_set<int> &level, unordered_map<int,int> &color, int curCol)
{

    vector<int> newlevel;
    for(auto i: level){
        for(auto j: adj[i]){
            if(color.find(j) == color.end()){
                newlevel.push_back(j);
                
            }
            color[j] = curCol;
        }
    }
    
    int* recvSizes = new int[nProcs];
    int si = newlevel.size();
    MPI_Allgather(&si, 1, MPI_INT, recvSizes, 1, MPI_INT, MPI_COMM_WORLD);

    int* recvDispls = new int[nProcs];
    recvDispls[0] = 0;
    for(int i = 1 ; i < nProcs ; i ++){
        recvDispls[i] = recvDispls[i-1] + recvSizes[i-1];
    }
    int* recvbuf = new int[recvDispls[nProcs-1] + recvSizes[nProcs-1]];
    int ns = newlevel.size();
    MPI_Allgatherv(newlevel.data(), ns, MPI_INT, recvbuf, recvSizes, recvDispls, MPI_INT, MPI_COMM_WORLD);
    level.clear();
    for(int i = 0 ; i < recvDispls[nProcs-1] + recvSizes[nProcs-1] ; i ++){
        int u = recvbuf[i];
            if(myRank == parProc[u]) level.insert(u);
            color[u] = curCol;
    }
    
    
    delete[] recvSizes;
    delete[] recvDispls;
    newlevel.clear();

    int flag =1;
    if(level.size() == 0) flag = 0;

    int newflag;
    MPI_Allreduce(&flag, &newflag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    flag = newflag;
    if(flag == 0){
        return;
    }
    else{
        bfs(adj, nProcs, myRank, level, color, curCol);
    }
}



void produceUndirectedGraph(unordered_map<int,vector<int>> &adj, unordered_map<int,vector<int>> &adj2, int nProcs, int myRank){
    int *displs = new int[nProcs];
    int *recvbuf = new int[nProcs];
    int *sendbuf = new int[nProcs]; // contains the number of nodes to be sent to each process
    int* count = new int[nProcs];
    int* recvdispls = new int[nProcs];

    for(int i = 0 ; i < nProcs ; i ++){
        displs[i] = 0;
        recvbuf[i] = 0;
        sendbuf[i] = 0; 
        count[i] = 0;
        recvdispls[i] = 0;
    }

    for(auto i: adj){
        int u = i.first;
        for(auto j: adj[u]){
            sendbuf[parProc[j]]+=2;
        }
    }

    for(int i = 1 ; i < nProcs ; i ++){
        displs[i] = displs[i-1] + sendbuf[i-1];
    }

    MPI_Alltoall(sendbuf, 1, MPI_INT, recvbuf, 1, MPI_INT, MPI_COMM_WORLD);

    for(int i = 1 ; i < nProcs ; i ++){
        recvdispls[i] = recvdispls[i-1] + recvbuf[i-1];
    }

    int *sendbuf2 = new int[displs[nProcs-1] + sendbuf[nProcs-1]];

    for(auto i: adj){
        int u = i.first;
        for(auto j: adj[u]){
            int par = parProc[j];
            sendbuf2[displs[par] + count[par]] = j;
            sendbuf2[displs[par] + count[par] + 1] = u;
            count[par]+=2;
        }
    }

    int *recvbuf2 = new int[recvdispls[nProcs-1] + recvbuf[nProcs-1]];

    MPI_Alltoallv(sendbuf2, sendbuf, displs, MPI_INT, recvbuf2, recvbuf, recvdispls, MPI_INT, MPI_COMM_WORLD);

    adj2 = adj;
    for(int i = 0 ; i < recvdispls[nProcs-1] + recvbuf[nProcs-1] ; i += 2){
        int u = recvbuf2[i];
        int v = recvbuf2[i+1];
        if(adj2.find(u) == adj2.end()){
            adj2[u] = vector<int>();
        }
        adj2[u].push_back(v);
    }


}




int main(int argc, char* argv[]){


    Input input(argc, argv);
    string filename, headerpath, outputpath;
    int kstart, kend, verbose, taskid,p;

    if(input.findInArgv("--inputpath")){
        filename = input.getArg("--inputpath");
    }
    else{
        cout<<"NO FILENAME GIVEN"<<endl;
        return 0;
        filename = "A2/test3/test-input-3.gra";
    }

    if(input.findInArgv("--outputpath")){
        outputpath = input.getArg("--outputpath");
    }
    else{
        cout<<"OUTPUTFILE NAME NOT GIVEN"<<endl;
        return 0;
    }



    if(input.findInArgv("--endk")){
        kend = stoi(input.getArg("--endk"));
    }
    else{
        cout<<"KEND NOT GIVEN"<<endl;
        return 0;
    }

    if(input.findInArgv("--headerpath")){
        headerpath = input.getArg("--headerpath");
    }
    else{
        cout<<"HEADERPATH NOT GIVEN"<<endl;
        return 0;
    }

    if(input.findInArgv("--verbose")){
        verbose = stoi(input.getArg("--verbose"));
    }
    else{
        verbose = 0;
    }

    if(input.findInArgv("--taskid")){
        taskid = stoi(input.getArg("--taskid"));
    }
    else{
        taskid = 1;
    }

    if(taskid == 1){

    if(input.findInArgv("--startk")){
        kstart = stoi(input.getArg("--startk"));
    }
    else{
        cout<<"KSTART NOT GIVEN"<<endl;
        return 0;
    }

    }

    if(taskid ==2 ){
        if(input.findInArgv("--p")){
            p = stoi(input.getArg("--p"));
        }
        else{
            cout<<"P NOT GIVEN"<<endl;
            return 0;
        }
    }

    int supported;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &supported);
    assert(MPI_THREAD_MULTIPLE == supported);
    int nProcs, myRank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    vector<int> offsets;
    int **supp;
    unordered_set<int> **triangles;
    unordered_map<int,vector<int> > undirAdj;
    unordered_map<int,vector<int> > dirAdj;
    unordered_map<int,vector<int> > adjDel;
    vector<int> degree;
    FILE *file = fopen(filename.c_str(), "rb");

    FILE *outputFile = fopen(outputpath.c_str(), "w");

    n = getw(file);
    m = getw(file);
    loadHeader(headerpath, offsets,degree, file);

    parProc = new int[n];
    int div = n/nProcs;
    for(int i = 0 ; i < n ; i ++){
        parProc[i] = i/div;
        if(parProc[i] >= nProcs){
            parProc[i] = nProcs-1;
        }
        if(parProc[i] == myRank){
            myNodes.push_back(i);
        }

    }
    initGraphNew(file,dirAdj, undirAdj, offsets, myRank,nProcs, degree, supp, triangles);
    int countEdges =0;
    int suppind =0;
    for(auto i:dirAdj){
        int a = i.first;
        adjDel[a] = vector<int>();
        for(auto j:dirAdj[a]){
            adjDel[a].push_back(0);
            countEdges++;
        }
    }

    MPI_Barrier( MPI_COMM_WORLD);
    int numTri = suppCalc(nProcs,myRank, dirAdj, supp, triangles, file, offsets, degree);
    int maxTri = 0;
    MPI_Allreduce( &numTri, &maxTri, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if(taskid ==1){
        for(int k = kstart; k<=kend; k++){
            int exist;
            exist = propLight(dirAdj,adjDel, degree, myRank, nProcs, k, supp, triangles,countEdges);
            
            ll ed =0;
            unordered_map<int,vector<int> > newAdj_temp;
            for(auto i:adjDel){
                int a = i.first;
                vector<int> temp;
                for(int j=0; j<adjDel[a].size(); j++){
                    if(adjDel[a][j]==0){
                        temp.push_back(dirAdj[a][j]);
                        ed++;
                    }
                }
                newAdj_temp[a] = temp;
            }
            adjDel.clear();
            dirAdj = newAdj_temp;
            newAdj_temp.clear();
            for(auto i:dirAdj){
                int a = i.first;
                adjDel[a] = vector<int>();
                for(auto j:dirAdj[a]){
                    adjDel[a].push_back(0);
                }
            }

            if(verbose == 1){
                if(myRank==0){
                    fprintf(outputFile, "%d\n", exist);
                }
                vector<int> nonempty;
                for(auto i:dirAdj){
                    if(i.second.size()!=0){
                        nonempty.push_back(i.first);
                    }
                }
                int *recvcounts = new int[nProcs];
                int *displs = new int[nProcs];
                int ne = nonempty.size();
                MPI_Allgather( &ne, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
                displs[0] = 0;
                for(int i = 1 ; i < nProcs ; i ++){
                    displs[i] = displs[i-1] + recvcounts[i-1];
                }
                int *nonemptyAll = new int[displs[nProcs-1]+recvcounts[nProcs-1]];
                MPI_Allgatherv(nonempty.data(), nonempty.size(), MPI_INT, nonemptyAll, recvcounts, displs, MPI_INT, MPI_COMM_WORLD);
                int *empty = new int[n];
                for(int i = 0 ; i < n ; i ++){
                    empty[i] = 1;
                }
                for(int i = 0 ; i < displs[nProcs-1]+recvcounts[nProcs-1] ; i ++){
                    empty[nonemptyAll[i]] = 0;
                }


                unordered_map<int, vector<int> > newAdj2;
                produceUndirectedGraph(dirAdj, newAdj2, nProcs, myRank);
                int curNode =0;
                unordered_set<int> level;
                unordered_map<int,int> color;
                int curCol = 1;
                while(curNode < n){
                    if(parProc[curNode]== myRank){
                        int flag =0;
                        if(color.find(curNode) == color.end() && empty[curNode]==0 ){ 
                            color[curNode] = curCol;
                            level.insert(curNode);
                            flag =1;
                            bfs(newAdj2,nProcs, myRank, level, color, curCol);
                            curCol++;
                        }

                    }
                    else{
                        int flag;
                        if(color.find(curNode) == color.end() && empty[curNode]==0 ){ 
                            color[curNode] = curCol;
                            bfs(newAdj2,nProcs, myRank, level, color, curCol);
                            curCol++;
                        }
                    }
                    curNode++;
                }
                int numCC = curCol-1;
            

                if(myRank==0){
                    vector<vector<int>> components(numCC);
                    for(auto i:color){
                        components[i.second-1].push_back(i.first);
                    }
                    if(numCC>0){
                        fprintf(outputFile, "%d\n", numCC);
                        for(auto i:components){
                            for(auto j:i){
                                if(j == i.back()){
                                    fprintf(outputFile, "%d", j);
                                }
                                else{
                                    fprintf(outputFile, "%d ", j);
                                }
                            
                            }
                            fprintf(outputFile, "\n");
                        }
                    }
                }

            }
            else{
                if(myRank == 0){
                    if(k == kend)
                        fprintf(outputFile, "%d", exist);
                    else
                        fprintf(outputFile, "%d ", exist);

                }
            }


        }
    }

    else if(taskid ==2)
    {

            int exist;
            exist = propLight(dirAdj,adjDel, degree, myRank, nProcs, kend, supp, triangles,countEdges);

            ll ed =0;
            unordered_map<int,vector<int> > newAdj_temp;
            for(auto i:adjDel){
                int a = i.first;
                vector<int> temp;
                for(int j=0; j<adjDel[a].size(); j++){
                    if(adjDel[a][j]==0){
                        temp.push_back(dirAdj[a][j]);
                        ed++;
                    }
                }
                newAdj_temp[a] = temp;
            }
            adjDel.clear();
            dirAdj = newAdj_temp;
            newAdj_temp.clear();
            for(auto i:dirAdj){
                int a = i.first;
                adjDel[a] = vector<int>();
                for(auto j:dirAdj[a]){
                    adjDel[a].push_back(0);
                }
            }
            if(verbose == 0){
                vector<int> emptyLocal;
                for(auto u: myNodes){
                    if(dirAdj[u].size()==0){
                        emptyLocal.push_back(u);
                    }
                }
                int *recvcounts = new int[nProcs];
                int *displs = new int[nProcs];
                int ne = emptyLocal.size();
                MPI_Allgather( &ne, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
                displs[0] = 0;
                for(int i = 1 ; i < nProcs ; i ++){
                    displs[i] = displs[i-1] + recvcounts[i-1];
                }
                int *emptyAll = new int[displs[nProcs-1]+recvcounts[nProcs-1]];
                MPI_Allgatherv(emptyLocal.data(), emptyLocal.size(), MPI_INT, emptyAll, recvcounts, displs, MPI_INT, MPI_COMM_WORLD);
                int *empty = new int[n];
                for(int i = 0 ; i < n ; i ++){
                    empty[i] = 0;
                }
                for(int i = 0 ; i < displs[nProcs-1]+recvcounts[nProcs-1] ; i ++){
                    empty[emptyAll[i]] = 1;
                }


                unordered_map<int, vector<int> > newAdj2;
                produceUndirectedGraph(dirAdj, newAdj2, nProcs, myRank);
                int curNode =0;
                unordered_set<int> level;
                unordered_map<int,int> color;
                int curCol = 1;
                while(curNode < n){
                    if(parProc[curNode]== myRank){
                        int flag =0;
                        if(color.find(curNode) == color.end() && empty[curNode]==0 ){ 
                            color[curNode] = curCol;
                            level.insert(curNode);
                            flag =1;
                            bfs(newAdj2,nProcs, myRank, level, color, curCol);
                            curCol++;
                        }

                    }
                    else{
                        int flag;
                        if(color.find(curNode) == color.end() && empty[curNode]==0 ){ 
                            color[curNode] = curCol;
                            bfs(newAdj2,nProcs, myRank, level, color, curCol);
                            curCol++;
                        }
                    }
                    curNode++;
                }
                
                int numCC = curCol-1;
                int *recvSizes = new int[nProcs];
                int cnt = 2*color.size();
                MPI_Allgather(&cnt, 1, MPI_INT, recvSizes, 1, MPI_INT, MPI_COMM_WORLD);
                int *displs2 = new int[nProcs];
                displs2[0] = 0;
                for(int i=1; i<nProcs; i++){
                    displs2[i] = displs2[i-1] + recvSizes[i-1];
                }
                int *recvBuf = new int[displs2[nProcs-1] + recvSizes[nProcs-1]];

                int *sendBuf = new int[2*color.size()];
                int ind = 0;
                for(auto i:color){
                    sendBuf[ind++] = i.first;
                    sendBuf[ind++] = i.second;
                }
                MPI_Allgatherv(sendBuf, 2*color.size(), MPI_INT, recvBuf, recvSizes, displs2, MPI_INT, MPI_COMM_WORLD);
                int sz = displs2[nProcs-1] + recvSizes[nProcs-1];
                vector<int> allCol(n,-1);
                for(int i=0; i<displs2[nProcs-1] + recvSizes[nProcs-1]; i+=2){
                    allCol[recvBuf[i]] = recvBuf[i+1];
                }
                vector<int> ans;
                #pragma omp parallel for
                for(int ni =0;ni<myNodes.size();ni++){
                    int a = myNodes[ni];
                    unordered_set<int> s;
                    for(auto b: undirAdj[a]){
                        if(allCol[b]!=-1){
                            s.insert(allCol[b]);
                        }
                    }
                    if(s.size()>=p){
                        #pragma omp critical
                        {
                            ans.push_back(a);
                        }
                    }
                }

                int size = ans.size();
                int allSize = 0;
                MPI_Reduce(&size, &allSize, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);   
                // send the answer to the root processor
                if(myRank == 0){
                    
                    int *recvSizes2 = new int[nProcs];
                    int cnt2 = ans.size();
                    MPI_Gather(&cnt2, 1, MPI_INT, recvSizes2, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    int *displs2 = new int[nProcs];
                    displs2[0] = 0;
                    for(int i=1; i<nProcs; i++){
                        displs2[i] = displs2[i-1] + recvSizes2[i-1];
                    }
                    int *recvBuf2 = new int[displs2[nProcs-1] + recvSizes2[nProcs-1]];
                    MPI_Gatherv(ans.data(), ans.size(), MPI_INT, recvBuf2, recvSizes2, displs2, MPI_INT, 0, MPI_COMM_WORLD);
                    fprintf(outputFile, "%d\n", allSize);
                    for(int i=0; i<displs2[nProcs-1] + recvSizes2[nProcs-1]; i++){
                        fprintf(outputFile, "%d ", recvBuf2[i]);
                    }
                }
                else{
                    MPI_Gather(&size, 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Gatherv(ans.data(), ans.size(), MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
                }

                         
                delete[] recvSizes;
                delete[] displs;
                delete[] recvBuf;
                delete[] sendBuf;

            }
            else if(verbose == 1){
                vector<int> emptyLocal;

                for(auto u: myNodes){
                    if(dirAdj[u].size()==0){
                        emptyLocal.push_back(u);
                    }
                }
                int *recvcounts = new int[nProcs];
                int *displs = new int[nProcs];
                int ne = emptyLocal.size();
                MPI_Allgather( &ne, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
                displs[0] = 0;
                for(int i = 1 ; i < nProcs ; i ++){
                    displs[i] = displs[i-1] + recvcounts[i-1];
                }
                int *emptyAll = new int[displs[nProcs-1]+recvcounts[nProcs-1]];
                MPI_Allgatherv(emptyLocal.data(), emptyLocal.size(), MPI_INT, emptyAll, recvcounts, displs, MPI_INT, MPI_COMM_WORLD);
                int *empty = new int[n];
                for(int i = 0 ; i < n ; i ++){
                    empty[i] = 0;
                }
                for(int i = 0 ; i < displs[nProcs-1]+recvcounts[nProcs-1] ; i ++){
                    empty[emptyAll[i]] = 1;
                }


                unordered_map<int, vector<int> > newAdj2;

                produceUndirectedGraph(dirAdj, newAdj2, nProcs, myRank);
                
                int curNode =0;
                unordered_set<int> level;
                unordered_map<int,int> color;
                int curCol = 1;
                while(curNode < n){
                    if(parProc[curNode]== myRank){
                        int flag =0;
                        if(color.find(curNode) == color.end() && empty[curNode]==0 ){ 
                            color[curNode] = curCol;
                            level.insert(curNode);
                            flag =1;
                            bfs(newAdj2,nProcs, myRank, level, color, curCol);
                            curCol++;
                        }

                    }
                    else{
                        int flag;
                        if(color.find(curNode) == color.end() && empty[curNode]==0 ){ 
                            color[curNode] = curCol;
                            bfs(newAdj2,nProcs, myRank, level, color, curCol);
                            curCol++;
                        }
                    }
                    curNode++;
                }
                
                int numCC = curCol-1;
                int *recvSizes = new int[nProcs];
                int cnt = 2*color.size();
                MPI_Allgather(&cnt, 1, MPI_INT, recvSizes, 1, MPI_INT, MPI_COMM_WORLD);
                int *displs2 = new int[nProcs];
                displs2[0] = 0;
                for(int i=1; i<nProcs; i++){
                    displs2[i] = displs2[i-1] + recvSizes[i-1];
                }
                int *recvBuf = new int[displs2[nProcs-1] + recvSizes[nProcs-1]];

                int *sendBuf = new int[2*color.size()];
                int ind = 0;
                for(auto i:color){
                    sendBuf[ind++] = i.first;
                    sendBuf[ind++] = i.second;
                }
                MPI_Allgatherv(sendBuf, 2*color.size(), MPI_INT, recvBuf, recvSizes, displs2, MPI_INT, MPI_COMM_WORLD);
                int sz = displs2[nProcs-1] + recvSizes[nProcs-1];
                vector<int> allCol(n,-1);
                for(int i=0; i<displs2[nProcs-1] + recvSizes[nProcs-1]; i+=2){
                    allCol[recvBuf[i]] = recvBuf[i+1];
                }
                vector<int> influencerVertex;
                vector<int> influencedVertex;
                #pragma omp parallel
                {
                for(int ni =0;ni<myNodes.size();ni++){
                    #pragma omp single
                    {
                    int a = myNodes[ni];
                    unordered_set<int> s;
                    for(auto b: undirAdj[a]){
                        if(allCol[b]!=-1){
                            s.insert(allCol[b]);
                        }
                    }
                    if(s.size()>=p){
                        #pragma omp critical
                        {
                            influencerVertex.push_back(a);
                            influencedVertex.push_back(a);
                            influencedVertex.push_back(s.size());
                            for(auto b:s){
                                influencedVertex.push_back(b);
                            }
                        }
                    }
                    }
                }
                }

                int size = influencerVertex.size();
                int sizeInfluenced = influencedVertex.size();
                int allSize = 0;
                MPI_Reduce(&size, &allSize, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);   
                int *recvInfluencer;
                int *recvInfluenced;
                if(myRank == 0){
                    int *recvSizes2 = new int[nProcs];
                    int cnt2 = influencerVertex.size();
                    MPI_Gather(&cnt2, 1, MPI_INT, recvSizes2, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    int *displs2 = new int[nProcs];
                    displs2[0] = 0;
                    for(int i=1; i<nProcs; i++){
                        displs2[i] = displs2[i-1] + recvSizes2[i-1];
                    }
                    recvInfluencer = new int[displs2[nProcs-1] + recvSizes2[nProcs-1]];
                    MPI_Gatherv(influencerVertex.data(), influencerVertex.size(), MPI_INT, recvInfluencer, recvSizes2, displs2, MPI_INT, 0, MPI_COMM_WORLD);

                    int *recvSizes3 = new int[nProcs];
                    int cnt3 = influencedVertex.size();
                    MPI_Gather(&cnt3, 1, MPI_INT, recvSizes3, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    int *displs3 = new int[nProcs];
                    displs3[0] = 0;
                    for(int i=1; i<nProcs; i++){
                        displs3[i] = displs3[i-1] + recvSizes3[i-1];
                    }
                    int finInfSize = displs3[nProcs-1] + recvSizes3[nProcs-1];
                    recvInfluenced = new int[displs3[nProcs-1] + recvSizes3[nProcs-1]];
                    MPI_Gatherv(influencedVertex.data(), influencedVertex.size(), MPI_INT, recvInfluenced, recvSizes3, displs3, MPI_INT, 0, MPI_COMM_WORLD);

                    
                    vector<vector<int>> components(numCC);
                    for(auto i:color){
                        components[i.second-1].push_back(i.first);
                    }


                    fprintf(outputFile, "%d\n", allSize);
                    if(allSize>0){
                        int i =0;
                        while(i<finInfSize){
                            int influencer = recvInfluenced[i++];
                            int size = recvInfluenced[i++];
                            fprintf(outputFile, "%d\n", influencer);
                            for(int j=0; j<size; j++){
                                for(auto k: components[recvInfluenced[i]-1]){
                                    fprintf(outputFile, "%d ", k);
                                }
                                i++;

                            }
                            fprintf(outputFile, "\n");
                        }
                    }
                }
                else{
                    MPI_Gather(&size, 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Gatherv(influencerVertex.data(), influencerVertex.size(), MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Gather(&sizeInfluenced, 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Gatherv(influencedVertex.data(), influencedVertex.size(), MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
                }
                delete[] recvSizes;
                delete[] displs;
                delete[] recvBuf;
                delete[] sendBuf;
            }
    }
    fclose(file);
    MPI_Finalize();
    return 0;
}